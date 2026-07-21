"""
Build portable Windows distribution (NO PyInstaller).

目的：完整复刻 source repo 的相对路径布局，让源码不经 PyInstaller 也能跑。
- 用户把文件夹解压到任何位置 → 双击 启动.bat → 即用
- 改 .py 之后只需要重新打包源码部分 (5s) → 不用 PyInstaller 等待
- 最终功能稳定后再走 build_exe.py 出 .exe (部署给不动 Python 的用户)

布局（相对路径，全部基于 bat 所在目录 %~dp0）：
    <ROOT>/                         ← bat 解压位置 (任意路径)
    ├── 启动.bat                     ← 双击启动
    ├── README.txt
    └── _internal/
        ├── python/                  ← 嵌入式 Python 3.10
        ├── site-packages/           ← pip install --target (deps)
        └── src/                     ← 源码 + 真实权重 (保留 source repo 布局)
            ├── scripts/
            │   └── gui/
            │       ├── coil_tip_viz_gui.py
            │       ├── hyper_inference.py
            │       └── framediff/...
            └── runs/
                ├── coil_panet_ablation/v26_mid_strong_full_300ep/weights/best.pt
                └── deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# ----- 嵌入式 Python 3.10 (win-amd64, 7.4MB zip) -----
PY_VERSION = "3.10.11"
PY_EMBED_URL = f"https://www.python.org/ftp/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip"
# 清华镜像 (国内 CI 用)
PY_EMBED_MIRROR = f"https://mirrors.tuna.tsinghua.edu.cn/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip"

# 部署依赖
DEPS = [
    "ultralytics==8.0.227",
    "opencv-python-headless",   # headless 省空间（GUI 不需要 cv2 显示）
    "pillow",
    "av",
    "ttkbootstrap",
]


def find_weights() -> Path:
    """真实 v26 best.pt 优先，找不到 fallback v18.3，仍然没有就 fail。"""
    candidates = [
        REPO_ROOT / "runs" / "coil_panet_ablation" / "v26_mid_strong_full_300ep" / "weights" / "best.pt",
        REPO_ROOT / "runs" / "deploy_best" / "v18_3_epoch60_hard_neg_weak_aug.pt",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1024 * 1024:
            print(f"[OK] Weights: {p}  ({p.stat().st_size//1024//1024} MB)")
            return p
    raise FileNotFoundError(
        "No real weights (≥1MB) found. Abort — placeholder would make GUI crash on startup."
    )


def download_python_embed(dest: Path) -> None:
    """下载并解压嵌入式 Python 到 _internal/python/"""
    if (dest / "python.exe").exists():
        print(f"[SKIP] Python embed already at {dest}")
        return
    print(f"[1/6] Downloading Python {PY_VERSION} embed (~7.4MB)...")
    tmp_zip = dest.parent / "_python_embed.zip"
    # 优先清华镜像，失败回落官方源
    for url in (PY_EMBED_MIRROR, PY_EMBED_URL):
        try:
            urllib.request.urlretrieve(url, tmp_zip)
            print(f"  ✓ from {url}")
            break
        except Exception as e:
            print(f"  ⚠ {url} failed: {e}")
    else:
        raise RuntimeError("Failed to download Python embed")
    with zipfile.ZipFile(tmp_zip) as zf:
        zf.extractall(dest)
    tmp_zip.unlink()


def install_deps(py_exe: Path, target: Path) -> None:
    """用嵌入 Python 的 pip 安装 deps 到 target/（不污染系统）"""
    print(f"[2/6] pip install --target={target}")
    print(f"      deps: {DEPS}")
    subprocess.run([str(py_exe), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run(
        [str(py_exe), "-m", "pip", "install", "--target", str(target)] + DEPS,
        check=True,
    )
    # 列出安装结果
    for pkg in ("ultralytics", "torch", "cv2", "PIL", "av", "ttkbootstrap"):
        try:
            subprocess.run(
                [str(py_exe), "-c", f"import {pkg}; print(f'  ✓ {pkg} ' + (__import__('importlib.metadata').metadata.version({pkg!r}) if __import__('importlib').util.find_spec({pkg!r}) else '?'))"],
                check=False, capture_output=True, text=True,
            )
        except Exception:
            pass


def copy_source(dist: Path) -> None:
    """拷贝源码到 _internal/src/scripts/gui/ 保持相对路径完整"""
    print("[3/6] Copying source files (relative paths preserved)")
    src_gui = dist / "_internal" / "src" / "scripts" / "gui"
    src_gui.mkdir(parents=True, exist_ok=True)
    for f in ("coil_tip_viz_gui.py", "hyper_inference.py", "frame_diff_wrapper.py",
              "build_portable.py"):
        src = SCRIPT_DIR / f
        if src.exists():
            shutil.copy(src, src_gui / f)
    # framediff 子目录完整复制
    shutil.copytree(SCRIPT_DIR / "framediff", src_gui / "framediff", dirs_exist_ok=True)
    print(f"  ✓ scripts/gui/*.py + framediff/")


def copy_weights(dist: Path) -> None:
    """拷贝真实权重到 _internal/src/runs/ (保持 runs/... 相对路径)"""
    print("[4/6] Copying real weights (v26 SOTA + v18.3 legacy fallback)")
    runs_dst = dist / "_internal" / "src" / "runs"
    runs_dst.mkdir(parents=True, exist_ok=True)
    cp = REPO_ROOT / "runs" / "coil_panet_ablation" / "v26_mid_strong_full_300ep"
    if cp.exists():
        shutil.copytree(cp, runs_dst / "coil_panet_ablation" / "v26_mid_strong_full_300ep",
                        dirs_exist_ok=True)
        bp = runs_dst / "coil_panet_ablation" / "v26_mid_strong_full_300ep" / "weights" / "best.pt"
        print(f"  ✓ {bp.relative_to(dist)}  ({bp.stat().st_size//1024//1024} MB)")
    legacy = REPO_ROOT / "runs" / "deploy_best"
    if legacy.exists():
        shutil.copytree(legacy, runs_dst / "deploy_best", dirs_exist_ok=True)
    # 任何 data/coil/*.yaml 也带上
    data_src = REPO_ROOT / "data" / "coil"
    if data_src.exists():
        data_dst = dist / "_internal" / "src" / "data" / "coil"
        data_dst.mkdir(parents=True, exist_ok=True)
        for y in data_src.glob("*.yaml"):
            shutil.copy(y, data_dst / y.name)


def write_pth(python_dir: Path) -> None:
    """写 python310._pth 让嵌入 Python 找到 site-packages 和 源码相对路径"""
    pth = python_dir / f"python{PY_VERSION.replace('.', '')[:3]}._pth"
    # 用 python.exe 命令推断版本号部分
    pth_file = python_dir / "python310._pth"
    pth_file.write_text(
        "python310.zip\n"
        ".\n"
        "..\\_internal\\src\\scripts\\gui\n"
        "..\\_internal\\src\\scripts\\gui\\framediff\n"
        "..\\_internal\\site-packages\n"
        "..\\_internal\\src\n"
        "import site\n",
        encoding="utf-8",
    )
    print(f"  ✓ {pth_file.name} (sys.path += ../_internal/src/scripts/gui + site-packages)")


def write_launcher(dist: Path) -> None:
    """写 启动.bat 和 README.txt"""
    print("[5/6] Writing launcher + README")
    launcher = dist / "启动.bat"
    launcher.write_text(
        "@echo off\r\n"
        "title Coil Tip Detection GUI (Portable)\r\n"
        "cd /d %~dp0\r\n"
        "set \"ROOT=%cd%\"\r\n"
        "set \"PYTHONIOENCODING=utf-8\"\r\n"
        "set \"PYTHONPATH=%ROOT%\\_internal\\src\\scripts\\gui;%ROOT%\\_internal\\src\\scripts\\gui\\framediff;%ROOT%\\_internal\\site-packages;%PYTHONPATH%\"\r\n"
        "\"%ROOT%\\_internal\\python\\python.exe\" \"%ROOT%\\_internal\\src\\scripts\\gui\\coil_tip_viz_gui.py\"\r\n"
        "if errorlevel 1 pause\r\n",
        encoding="utf-8",
    )
    readme = dist / "README.txt"
    readme.write_text(
        "============================================================\r\n"
        " Coil Tip Detection GUI - Portable Edition\r\n"
        "============================================================\r\n"
        "\r\n"
        " Quick start:\r\n"
        "   1. 解压到任意目录 (C:\\Tools\\, D:\\Work\\, Desktop ...)\r\n"
        "   2. 双击 启动.bat\r\n"
        "\r\n"
        " 文件结构:\r\n"
        "   _internal\\python\\         = 嵌入式 Python 3.10 (无需安装)\r\n"
        "   _internal\\site-packages\\  = ultralytics + opencv + pillow + av + ttkbootstrap\r\n"
        "   _internal\\src\\            = 源码 + v26/v18.3 真实权重\r\n"
        "   启动.bat                  = 双击启动\r\n"
        "\r\n"
        " 系统需求:\r\n"
        "   - Windows 10/11 (64-bit)\r\n"
        "   - VC++ Redistributable (Win10/11 自带)\r\n"
        "   - NVIDIA GPU + 驱动 (可选, 没有走 CPU)\r\n"
        "\r\n"
        " 完全不需要 Python / Anaconda / ultralytics / torch 任何环境!\r\n",
        encoding="utf-8",
    )


def make_zip(dist: Path) -> Path:
    """压缩 _internal/ 和 启动.bat / README.txt 成 zip"""
    print("[6/6] Zipping artifact")
    parent = dist.parent
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=REPO_ROOT, capture_output=True, text=True, check=True
                             ).stdout.strip()
    except Exception:
        sha = "local"
    zip_name = f"CoilTipViz-portable-{sha}.zip"
    zip_path = parent / zip_name
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, _, files in os.walk(dist):
            for f in files:
                p = Path(root) / f
                arcname = p.relative_to(parent).as_posix()  # CoilTipViz/...
                zf.write(p, arcname)
    print(f"  ✓ {zip_name}  size={zip_path.stat().st_size//1024//1024} MB")
    return zip_path


def main() -> int:
    dist = SCRIPT_DIR / "dist" / "CoilTipViz"
    if dist.exists():
        print(f"[CLEAN] Removing old dist/CoilTipViz")
        shutil.rmtree(dist)
    (dist / "_internal" / "python").mkdir(parents=True)
    (dist / "_internal" / "site-packages").mkdir(parents=True)
    (dist / "_internal" / "src").mkdir(parents=True)

    download_python_embed(dist / "_internal" / "python")
    py_exe = dist / "_internal" / "python" / "python.exe"
    write_pth(dist / "_internal" / "python")
    install_deps(py_exe, dist / "_internal" / "site-packages")
    copy_source(dist)
    copy_weights(dist)
    write_launcher(dist)
    zip_path = make_zip(dist)
    print()
    print("=" * 60)
    print(f" Build complete: {zip_path}")
    print(f" 测试: 双击 {dist / '启动.bat'}")
    print(f" 路径: {dist}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
