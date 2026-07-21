"""
Build a slim ZIP for users on route B (source + bat, no PyInstaller, no torch).

What's in the zip (~10 MB):
    CoilTipViz/
        coil_tip_viz_gui.py        (GUI entry)
        hyper_inference.py         (detector, with stdout patch)
        frame_diff_wrapper.py      (optional frame diff pipeline)
        frame_diff/                (frame diff internals)
            frame_diff_detector.py
            change_capture.py
            pyav_reader.py
        weights/best.pt            (v26 SOTA, 8 MB)
        install_hyper_yolo.bat     (Anaconda + GPU torch install)
        run_windows.bat            (double-click launcher)
        README.txt                 (quick start)

What is NOT in the zip:
    - Python interpreter       → user installs via Anaconda
    - ultralytics / torch      → installed by install_hyper_yolo.bat (~5 min, GPU torch ~2.5 GB)
    - cv2 / av / PIL / ...     → installed by install_hyper_yolo.bat

Why route B vs route A (.exe):
    - User has NVIDIA GPU: route B installs GPU torch → inference fast, route A bundles CPU torch
    - Bundle size: route A ~291 MB always, route B ~10 MB
    - Update iteration: route A full re-download, route B only changes re-downloaded (10 MB)

Usage:
    python build_src_zip.py            # outputs CoilTipViz-source.zip in dist/
    python build_src_zip.py /tmp/x/    # outputs at custom path
"""
from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
ZIP_BASENAME = "CoilTipViz-source"

# ---- Source layout inside the zip ----
# (src_relative_to_repo_path, dest_relative_to_zip_root)
ENTRIES: list[tuple[Path, str]] = [
    (Path("scripts/gui/coil_tip_viz_gui.py"), "CoilTipViz/coil_tip_viz_gui.py"),
    (Path("scripts/gui/hyper_inference.py"), "CoilTipViz/hyper_inference.py"),
    (Path("scripts/gui/frame_diff_wrapper.py"), "CoilTipViz/frame_diff_wrapper.py"),
    (Path("scripts/gui/framediff/frame_diff_detector.py"), "CoilTipViz/framediff/frame_diff_detector.py"),
    (Path("scripts/gui/framediff/change_capture.py"), "CoilTipViz/framediff/change_capture.py"),
    (Path("scripts/gui/framediff/pyav_reader.py"), "CoilTipViz/framediff/pyav_reader.py"),
    (Path("scripts/gui/install_hyper_yolo.bat"), "CoilTipViz/install_hyper_yolo.bat"),
    (Path("scripts/gui/run_windows.bat"), "CoilTipViz/run_windows.bat"),
    (Path("scripts/gui/build_exe.bat"), "CoilTipViz/build_exe.bat"),
    (Path("runs/coil_panet_ablation/v26_mid_strong_full_300ep/weights/best.pt"),
     "CoilTipViz/weights/best.pt"),
]


def main(out_dir: Path | None = None) -> int:
    out_dir = out_dir or SCRIPT_DIR / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{ZIP_BASENAME}.zip"

    # ---- Validate every source file exists & non-empty ----
    missing: list[Path] = []
    for repo_rel, _ in ENTRIES:
        full = REPO_ROOT / repo_rel
        if not full.is_file() or full.stat().st_size == 0:
            missing.append(full)
    if missing:
        print("[ERROR] Missing or empty source files:")
        for m in missing:
            print(f"  {m}")
        return 1

    # ---- Build zip ----
    if zip_path.exists():
        zip_path.unlink()
    total = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1) all entries above
        for repo_rel, dest in ENTRIES:
            full = REPO_ROOT / repo_rel
            sz = full.stat().st_size
            print(f"  + {dest}  ({sz // 1024} KB)")
            zf.write(full, dest)
            total += sz

        # 2) README at root
        readme = (
            "CoilTipViz — 钢卷头尾检测 (路线 B: GPU torch 安装)\n"
            "============================================\n\n"
            "Quick start (3 steps, total ~10 min):\n"
            "\n"
            "  1. 双击 install_hyper_yolo.bat (一次性, 装 Anaconda hyper-yolo env + GPU torch 2.5.1 cu121)\n"
            "\n"
            "  2. 双击 run_windows.bat 启动 GUI\n"
            "\n"
            "  3. 选择视频 → 开始检测\n"
            "\n"
            "\n"
            "Requirements:\n"
            "  - Windows 10/11 64-bit\n"
            "  - NVIDIA GPU + 驱动 (装 GPU 版 torch 用)\n"
            "  - 本文件 ~10 MB; torch 安装后占 ~5 GB\n"
            "\n"
            "Files:\n"
            "  CoilTipViz/coil_tip_viz_gui.py     - GUI 主程序\n"
            "  CoilTipViz/hyper_inference.py      - 模型推理 (含 stdout 兼容层)\n"
            "  CoilTipViz/framediff/              - 帧差法管线 (可选)\n"
            "  CoilTipViz/weights/best.pt         - v26 SOTA 部署权重 (8 MB, F1=0.9359)\n"
            "  install_hyper_yolo.bat             - 首次安装 (一次性, GPU torch)\n"
            "  run_windows.bat                    - 双击启动\n"
            "\n"
            "路线 A vs B:\n"
            "  A (.exe): 单文件, 不用装 Python/ultralytics/torch, 但要每次重新下载 291 MB; 推 CPU\n"
            "  B (源码 zip, 这个): 首次装 torch 慢, 但只用下 10 MB zip; 走 GPU 推, 速度快\n"
        )
        zf.writestr("CoilTipViz/README.txt", readme.encode("utf-8"))

    sz = zip_path.stat().st_size
    print(f"\n[OK] Built: {zip_path}  ({sz // 1024 // 1024} MB, source total {total // 1024 // 1024} MB)")
    return 0


if __name__ == "__main__":
    extra = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(Path(extra) if extra else None))
