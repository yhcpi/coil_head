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
    """Locate deploy weights in priority order:

    1. v26 mid-strong full 300ep best.pt (F1=0.9359 SOTA, 8MB)
    2. v18_3 hard neg weak aug pt (F1=0.9286 legacy SOTA, 32MB)
    3. placeholder.pt — REJECTED in real build, CI must force-import a real .pt
    """
    candidates = [
        REPO_ROOT / "runs" / "coil_panet_ablation" / "v26_mid_strong_full_300ep" / "weights" / "best.pt",
        REPO_ROOT / "runs" / "deploy_best" / "v18_3_epoch60_hard_neg_weak_aug.pt",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1024 * 1024:
            print(f"[OK] Weights: {p}  ({p.stat().st_size//1024//1024} MB)")
            return p
    # 真实权重一个都没有 → 失败，不打包 placeholder（否则 .exe 启动 ultralytics 必崩）
    raise FileNotFoundError(
        "No real weights found (≥1MB required).\n"
        f"  Searched:\n"
        f"    - {REPO_ROOT / 'runs' / 'coil_panet_ablation' / 'v26_mid_strong_full_300ep' / 'weights' / 'best.pt'}\n"
        f"    - {REPO_ROOT / 'runs' / 'deploy_best' / 'v18_3_epoch60_hard_neg_weak_aug.pt'}\n"
        "  Placeholder (0-byte) is NOT bundled — would cause 'NoneType has no attribute encoding'."
    )


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

    # ---- Step 4: Locate weights ----
    weights = find_weights()

    # ---- Step 5: Build argv ----
    framediff_dir = SCRIPT_DIR / "framediff"
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
        "--paths", str(SCRIPT_DIR),
        "--paths", str(framediff_dir),
        "--add-data", f"{SCRIPT_DIR / 'hyper_inference.py'};.",
        "--add-data", f"{SCRIPT_DIR / 'frame_diff_wrapper.py'};.",
        "--add-data", f"{framediff_dir / 'frame_diff_detector.py'};framediff",
        "--add-data", f"{framediff_dir / 'change_capture.py'};framediff",
        "--add-data", f"{framediff_dir / 'pyav_reader.py'};framediff",
        "--add-data", f"{weights};weights",
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