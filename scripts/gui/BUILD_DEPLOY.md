# Coil Tip Detection GUI — Build & Deploy Guide

## Zero-Dependency Deployment

**Goal**: end users double-click `CoilTipViz.exe` and it runs. **No Python, Anaconda, ultralytics, or torch needed on their PC.**

## One-Time Build (on YOUR Windows dev machine)

You only need to do this ONCE. After that, distribute the same `CoilTipViz/` folder to anyone.

### Prerequisites (one-time setup on your dev PC)

1. **Windows 10/11 (64-bit)**
2. **Anaconda** installed at one of these paths:
   - `D:\ProgramData\anaconda3\` (your case)
   - `C:\ProgramData\anaconda3\`
   - `%USERPROFILE%\anaconda3\`
3. **hyper-yolo env** with ultralytics (one-time):
   ```bat
   :: Open Anaconda Prompt
   conda create -n hyper-yolo python=3.10 -y
   conda activate hyper-yolo
   pip install ultralytics==8.0.227 opencv-python pillow pyav
   ```

### Build steps

1. Copy `scripts/gui/` folder to Windows (e.g. `D:\gui\`)
2. **Double-click `D:\gui\build_exe.bat`**
3. Wait 5-10 minutes for PyInstaller
4. Output: `D:\gui\dist\CoilTipViz\` (~1.5 GB)

That's it. Now distribute `dist\CoilTipViz\` to anyone.

## Distribution

### Package the folder

```bat
:: Right-click dist\CoilTipViz\ -> Send to -> Compressed (zipped) folder
:: Result: CoilTipViz.zip (~1.5 GB)
```

### Send to end users

- Email attachment (if <2GB), OneDrive, USB stick, LAN share, etc.
- User unzips to any folder, e.g. `D:\CoilTipViz\`
- User **double-clicks `CoilTipViz.exe`** — GUI appears

## End User Requirements (target PC)

| Requirement | Required? | Notes |
|---|---|---|
| Windows 10/11 (64-bit) | YES | Win10 1809+ or Win11 |
| VC++ Redistributable | YES | Preinstalled on 99% of PCs. If missing: https://aka.ms/vs/17/release/vc_redist.x64.exe |
| Python | NO | Bundled in _internal\ |
| Anaconda | NO | Not needed |
| ultralytics | NO | Bundled |
| torch (1GB) | NO | Bundled |
| NVIDIA GPU | NO (optional) | CPU mode works but ~5-10x slower |
| Admin rights | NO | Runs in user space |
| Internet | NO | Fully offline after install |

## File layout of distributable

```
CoilTipViz\                          <- ZIP this whole folder (~1.5 GB)
├── CoilTipViz.exe                   <- double-click to launch
├── 启动.bat                         <- alternate launcher (Chinese name)
├── README.txt                       <- end-user guide (in folder)
└── _internal\                       <- DO NOT touch, contains all dependencies
    ├── CoilTipViz.exe.manifest
    ├── python312.dll                <- embedded Python interpreter
    ├── torch\                       <- 1 GB ML framework
    ├── ultralytics\                 <- YOLO library
    ├── cv2\                         <- OpenCV
    ├── av\                          <- pyav video I/O
    ├── PIL\                         <- Pillow
    ├── weights\
    │   └── v18_3_epoch60_hard_neg_weak_aug.pt  <- default model
    ├── framediff\
    │   ├── frame_diff_detector.py
    │   ├── change_capture.py
    │   └── pyav_reader.py
    ├── hyper_inference.py
    ├── frame_diff_wrapper.py
    └── ... (200+ DLLs and .py files)
```

## Updating the model or code

1. Edit source code (e.g. swap `v18_3` → `v26` weights, or modify GUI)
2. Re-run `build_exe.bat`
3. Redistribute new `dist\CoilTipViz\` folder

Users replace their old `CoilTipViz\` with the new one. **No uninstall needed** — old `dist\` folder can be deleted.

## Troubleshooting build issues

| Symptom | Cause | Fix |
|---|---|---|
| `[ERROR] hyper-yolo env not found` | Anaconda not in standard path | Edit build_exe.bat, add your path to `for %%P in` |
| `ModuleNotFoundError: cv2` during PyInstaller | opencv not in hyper-yolo env | `pip install opencv-python` in hyper-yolo env |
| PyInstaller hangs | Antivirus scanning temp files | Add `dist/` and `build/` to AV exclusions |
| `.exe` > 2 GB | OneDir mode + torch | Expected. Use 7z / RAR for better compression |
| User reports "MSVCP140.dll missing" | VC++ not installed | Install VC++ Redistributable (link above) |

## Why not ONNX Runtime or pure C++?

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **PyInstaller (chosen)** | One command, no code change, full GUI | 1.5 GB | **Best for v1** |
| ONNX Runtime | Smaller (50 MB), faster startup | Still needs Python + onnxruntime; rewrite hyper_inference.py | Future v2 |
| Pure C++ + Qt | True zero-dep, 80 MB | 2-3 weeks rewrite of GUI + inference | Future v3 (if commercial) |

## Distribution checklist

Before sending to end users:

- [ ] `CoilTipViz.exe` double-click launches GUI on a clean Win10 VM
- [ ] Bundle includes at least one working `.pt` model
- [ ] `README.txt` in folder explains double-click usage
- [ ] Test on a PC WITHOUT Python installed
- [ ] Test on a PC WITHOUT NVIDIA GPU (CPU mode)
- [ ] Test on a PC with antivirus (add exclusions if flagged)