"""
PyInstaller wrapper - called by build_exe.bat or GitHub Actions.

Avoids all cmd.exe quoting / continuation issues by building argv in Python
where quoting is explicit.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # scripts/gui/ -> scripts/ -> repo root

print(f"[build_exe.py] script_dir={SCRIPT_DIR}")
print(f"[build_exe.py] repo_root={REPO_ROOT}")


def find_weights() -> Path:
    """Locate deploy SOTA weight (only v26, no legacy/fallback chain).

    Returns the single real .pt that will be bundled into the .exe.
    """
    primary = (
        REPO_ROOT / "runs" / "coil_panet_ablation" / "v26_mid_strong_full_300ep" / "weights" / "best.pt"
    )
    if primary.exists() and primary.stat().st_size > 1024 * 1024:
        print(f"[OK] Weights: {primary}  ({primary.stat().st_size//1024//1024} MB)")
        return primary
    raise FileNotFoundError(
        f"v26 deploy SOTA weight missing or too small:\n"
        f"  {primary}\n\n"
        f"Either:\n"
        f"  - Run scripts/run_v26_mid_strong_full_300ep.sh and copy best.pt to runs/coil_panet_ablation/v26_mid_strong_full_300ep/weights/\n"
        f"  - Or run 'git add -f runs/coil_panet_ablation/.../best.pt' to make sure it's tracked\n"
    )


def stage_weight_for_bundle(weights: Path) -> Path:
    """拷贝权重到 dist/best.pt (标准化名), PyInstaller 把它放到 _internal/weights/best.pt

    因为 PyInstaller 的 --add-data `src;dest/` 行为:
      - dest 有尾斜杠 → src 进入 dest/ 目录, 文件名保持原 basename
      - dest 无尾斜杠 → src 被重命名为 dest (单文件)
    把 src stage 成 best.pt → 用户代码统一找 _internal/weights/best.pt 即可。

    ⚠️ 文件名必须是 best.pt 而不是 _staged_best.pt: PyInstaller 保留 src 的 basename,
       之前误用 _staged_best.pt 导致 bundle 里出现 _internal/weights/_staged_best.pt (用户报错).
    """
    staged = SCRIPT_DIR / "dist" / "best.pt"
    staged.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(weights, staged)
    print(f"[OK] Staged weight for bundle: {staged}  → _internal/weights/best.pt")
    return staged


def main() -> int:
    # ---- Step 1: Find Python ----
    py = sys.executable
    print(f"[OK] Python: {py}")

    # ---- Step 2: Verify deps ----
    for mod in ("ultralytics", "cv2", "av", "PIL", "PyInstaller"):
        try:
            __import__(mod)
        except ImportError:
            print(f"[ERROR] missing module: {mod}")
            return 1
    print("[OK] All deps importable")

    # ---- Step 3: Clean old artifacts ----
    for d in ("build", "dist"):
        if (SCRIPT_DIR / d).exists():
            shutil.rmtree(SCRIPT_DIR / d)
            print(f"[OK] Cleaned {d}/")
    spec = SCRIPT_DIR / "CoilTipViz.spec"
    if spec.exists():
        spec.unlink()
        print("[OK] Cleaned CoilTipViz.spec")

    # ---- Step 4: Locate + stage weights for bundle ----
    weights_real = find_weights()
    weights_stage = stage_weight_for_bundle(weights_real)

    # ---- GPU inference step: detect CUDA torch in env, fail loud if not ----
    print("[5/6] Verifying CUDA torch available for PyInstaller bundle...")
    import importlib
    torch_mod = importlib.import_module("torch")
    if "+cu" not in torch_mod.__version__:
        print(f"[ERROR] torch version {torch_mod.__version__} is CPU-only!")
        print(f"        Want torch+cu121 (CUDA 12.1 wheel). Bundle would not include CUDA runtime.")
        print(f"        Re-run with: pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121")
        return 1
    print(f"[OK] torch={torch_mod.__version__}  cuda_available={torch_mod.cuda.is_available()}")
    if torch_mod.cuda.is_available():
        print(f"[OK] CUDA device: {torch_mod.cuda.get_device_name(0)}")
    else:
        print(f"[WARN] CUDA not available at build time (no NVIDIA runtime). Bundle should still include CUDA DLLs.")

    # ---- Step 5.5: Weight loading pre-flight (CRITICAL) ----
    # 模拟 .exe 启动时的真实 YOLO load, 如果 build 阶段就失败, 后续 PyInstaller 8 min 全白费
    # PyInstaller 不修改 .pt pickle 字节, 加载行为构建期 == 运行时
    #
    # ⚠️ v26 best.pt 用 Hyper-YOLO 仓库的 ultralytics 训练 (含自定义模块 MANet @ block.py:376),
    #    pickle 里 GLOBAL 指向 ultralytics.nn.modules.block.MANet. CI 装的是官方 PyPI ultralytics
    #    8.0.227 (无 MANet) → YOLO() 加载报 AttributeError: Can't get attribute 'MANet'.
    # 修复: pre-flight 前 sys.path 注入 repos/Hyper-YOLO (含扩展 ultralytics + MANet),
    #    让 MANet 类被 import → sys.modules 注册 → YOLO() 加载能找到.
    print("[5.5/6] Pre-flight: load staged .pt with ultralytics (catches all torch/weight bugs)")
    print(f"        staged: {weights_stage} ({weights_stage.stat().st_size//1024//1024} MB)")

    # --- Inject Hyper-YOLO ultralytics BEFORE pre-flight (so MANet is registered) ---
    hyper_yolo_root = REPO_ROOT / "repos" / "Hyper-YOLO"
    if hyper_yolo_root.exists():
        sys.path.insert(0, str(hyper_yolo_root))
        print(f"        sys.path 注入 Hyper-YOLO: {hyper_yolo_root} (含 MANet)")
    else:
        print(f"        [WARN] repos/Hyper-YOLO not found, MANet injection skipped (will likely fail pre-flight)")

    try:
        from ultralytics import YOLO
        _m = YOLO(str(weights_stage))
        print(f"[OK] YOLO() loaded at build time → runtime in .exe will load same way")
        del _m
    except Exception as _exc:
        print(f"[ERROR] Stage pre-flight FAIL: {type(_exc).__name__}: {_exc}")
        print(f"        Likely causes:")
        print(f"        - torch version mismatch (need exactly 2.5.1+cu121)")
        print(f"        - corrupt .pt file (re-run {weights_real.name} training)")
        print(f"        - ultralytics/torch dependency conflict (reinstall matching wheels)")
        print(f"        - MANet missing (Hyper-YOLO repo not injected) — check repos/Hyper-YOLO exists")
        return 1

    # ---- Step 5.6: Bundle dep sanity: ultralytics + cv2 + av + PIL + ttkbootstrap + tkinter ----
    print("[5.6/6] Pre-flight: every PyInstaller --collect-all target importable")
    for mod_name in ("cv2", "av", "PIL", "ttkbootstrap", "tkinter"):
        try:
            importlib.import_module(mod_name)
            print(f"[OK] {mod_name} importable")
        except ImportError as _e:
            print(f"[ERROR] {mod_name} NOT importable: {_e}")
            return 1

    # ---- Step 5: Build argv ----
    framediff_dir = SCRIPT_DIR / "framediff"
    # Hyper-YOLO 扩展 ultralytics 含 MANet, 必须打进 bundle, 否则 v26 best.pt 加载失败
    hyper_yolo_root = REPO_ROOT / "repos" / "Hyper-YOLO"
    argv = [
        py, "-m", "PyInstaller",
        "--noconfirm",
        "--noconsole",
        "--onedir",
        "--name", "CoilTipViz",
        "--collect-all", "ultralytics",
        "--collect-all", "torch",
        "--collect-all", "av",
        "--collect-all", "ttkbootstrap",          # TRAE IDE 风格 GUI 主题 (cosmo)
        "--hidden-import", "cv2",
        "--hidden-import", "PIL",
        "--hidden-import", "numpy",
        "--hidden-import", "tkinter",
        "--hidden-import", "tkinter.filedialog",
        "--hidden-import", "tkinter.messagebox",
        # ⚠️ v26 best.pt 引用 MANet (Hyper-YOLO 自定义模块), 必须显式 hidden-import + paths
        "--hidden-import", "ultralytics.nn.modules.block.MANet",
        "--paths", str(SCRIPT_DIR),
        "--paths", str(framediff_dir),
    ]
    if hyper_yolo_root.exists():
        # 把 Hyper-YOLO 仓库加入 PyInstaller 搜索路径, 包含扩展的 ultralytics/nn/modules/block.py
        argv += ["--paths", str(hyper_yolo_root)]
        print(f"        PyInstaller --paths 注入 Hyper-YOLO: {hyper_yolo_root}")
    argv += [
        "--add-data", f"{SCRIPT_DIR / 'hyper_inference.py'};.",
        "--add-data", f"{SCRIPT_DIR / 'frame_diff_wrapper.py'};.",
        "--add-data", f"{framediff_dir / 'frame_diff_detector.py'};framediff",
        "--add-data", f"{framediff_dir / 'change_capture.py'};framediff",
        "--add-data", f"{framediff_dir / 'pyav_reader.py'};framediff",
        "--add-data", f"{weights_stage};weights/",  # 尾斜杠 = 目录; basename stage 为 best.pt → _internal/weights/best.pt
        str(SCRIPT_DIR / "coil_tip_viz_gui.py"),
    ]

    # ---- Step 6: Run PyInstaller ----
    print("[3/4] PyInstaller packaging (5-10 minutes)...")
    print(f"  argv[0] (script): {argv[-1]}")
    print(f"  argv length: {len(argv)} args")
    result = subprocess.run(argv, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"[ERROR] PyInstaller failed with exit code {result.returncode}")
        return result.returncode

    # ---- Step 7: Create launcher bat and README ----
    print("[4/4] Creating launcher and README...")
    dist_dir = SCRIPT_DIR / "dist" / "CoilTipViz"
    dist_dir.mkdir(parents=True, exist_ok=True)

    launcher = dist_dir / "启动.bat"
    launcher.write_text(
        "@echo off\r\n"
        "title Coil Tip Detection GUI\r\n"
        "cd /d %~dp0\r\n"
        "CoilTipViz.exe\r\n"
        "if errorlevel 1 pause\r\n",
        encoding="utf-8",
    )

    readme = dist_dir / "README.txt"
    readme.write_text(
        "============================================================\r\n"
        " Coil Tip Detection GUI - End User Guide\r\n"
        "============================================================\r\n"
        "\r\n"
        " Quick start:\r\n"
        "   1. Double-click CoilTipViz.exe\r\n"
        "   2. Click 'Select Weight' - pick a .pt file (or use bundled default)\r\n"
        "   3. Click 'Select Video' - pick an .mp4 file\r\n"
        "   4. Click 'Start' button\r\n"
        "\r\n"
        " File list:\r\n"
        "   CoilTipViz.exe   - Main program, double-click to launch\r\n"
        "   启动.bat         - Same launcher, Chinese-named\r\n"
        "   _internal\\       - Dependency files, DO NOT DELETE\r\n"
        "\r\n"
        " System requirements:\r\n"
        "   - Windows 10/11 (64-bit)\r\n"
        "   - VC++ Redistributable (preinstalled on 99% of PCs)\r\n"
        "   - NVIDIA GPU + driver (optional, for GPU acceleration)\r\n"
        "\r\n"
        " No Python, Anaconda, ultralytics, or torch needed!\r\n",
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print(" Build complete!")
    print(f" Output: {dist_dir}")
    print("   CoilTipViz.exe   (double-click to launch)")
    print("   启动.bat         (alternate launcher)")
    print("   README.txt       (end-user guide, ship with .exe)")
    print("   _internal\\       (dependencies - DO NOT modify)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())