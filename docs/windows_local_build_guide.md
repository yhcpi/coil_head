# Windows 本地打包 CoilTipViz.exe — Step-by-Step

**适用场景**：你 Windows 机器有 NVIDIA GPU + driver，**不想走 GitHub Actions CI**，要本地直接跑 `build_exe.py` 出 exe。

**前情**：CI 路线 #18 失败，根因 = `repos/Hyper-YOLO` 没被 git track，CI runner checkout 后那个目录不存在，MANet 注入 fix 无效。已 commit `3383c24` 把 MANet class vendored 进 `scripts/gui/hyper_yolo_extensions/`（不依赖外部 repo）。本指南用 vendored 修复。

---

## 一次性准备（10-15 分钟）

### 1. 装 Python 3.10

```powershell
winget install Python.Python.3.10
# 或 python.org 下 .exe, 勾 "Add to PATH"
python --version    # 验证 3.10.x
```

### 2. 装 PyTorch CUDA 12.1 wheel（**关键：必须 +cu121，否则 build_exe.py 会拒绝打包**）

```powershell
python -m pip install --upgrade pip
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

python -m pip install "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" `
  --index-url https://download.pytorch.org/whl/cu121 `
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 2.5.1+cu121 True
```

### 3. 装其他依赖

```powershell
python -m pip install "ultralytics==8.0.227" opencv-python pillow av ttkbootstrap pyinstaller
```

### 4. 装 Microsoft Visual C++ Redistributable（PyInstaller bundle 需要）

```powershell
# 通常 Windows 11 已自带, 没的话:
winget install Microsoft.VCRedist.2015+.x64
```

### 5. 拉代码仓库

```powershell
cd D:\projects   # 你工作目录
git clone https://github.com/yhcpi/coil_head.git
cd coil_head
git log --oneline -3    # 验证 HEAD 是 3383c24 (vendored MANet)
```

### 6. 拉 v26 best.pt（如果 git 没 track 它）

```powershell
# best.pt 应该已经在仓库里, 检查:
dir runs\coil_panet_ablation\v26_mid_strong_full_300ep\weights\best.pt
# 如果文件不存在, 你需要从其他渠道拷过来 (本地训练或外部备份)
```

---

## 打包（8-12 分钟）

```powershell
cd D:\projects\coil_head\scripts\gui
python build_exe.py
```

**预期输出**：
```
[OK] Python: C:\...\python.exe
[OK] All deps importable
[OK] Weights: ...\best.pt  (7 MB)
[OK] Staged weight for bundle: ...\dist\best.pt
[OK] torch=2.5.1+cu121  cuda_available=True
[OK] CUDA device: NVIDIA GeForce RTX ...
hyper_yolo_extensions MANet vendored OK
[5.5/6] Pre-flight: load staged .pt with ultralytics
[OK] YOLO() loaded at build time → runtime in .exe will load same way
... PyInstaller ... (5-10 分钟)
Build complete!
Output: D:\projects\coil_head\scripts\gui\dist\CoilTipViz
```

**如果失败**：
- `torch version is CPU-only` → 你装了 CPU 版 torch, 重跑 step 2 装 +cu121
- `Stage pre-flight FAIL: Can't get attribute 'MANet'` → 你 clone 的不是 HEAD `3383c24`, `git pull` 后重试
- 其他 → 把错误贴给我看

---

## 出包路径

```
D:\projects\coil_head\scripts\gui\dist\CoilTipViz\
  CoilTipViz.exe             ← 主程序, 双击启动
  启动.bat                   ← 中文启动器 (可选)
  README.txt                 ← 给最终用户的简单说明
  _internal\                 ← PyInstaller bundle, 不要删
    weights\best.pt          ← v26 SOTA (8 MB)
    ultralytics\...          ← 含 MANet (vendored from 3383c24)
```

**双击 CoilTipViz.exe 即可启动**（无需 Python 环境）。

---

## 验证清单

打包完后建议你：
1. ✅ 双击 CoilTipViz.exe → 弹出 GUI 窗口
2. ✅ 用一张测试图 (eg `data/coil/images/val/*.jpg`) 跑推理 → 应有 bbox 输出
3. ✅ 选一个 .mp4 视频 → 跑 head/tail 检测 → 应有 bbox + TTA builtin
4. ✅ 命令行窗口无红色 ERROR (warnings OK)
5. ✅ GPU 使用率 > 0% (任务管理器 → 性能 → GPU)

---

## 替代路线（如果你想要更小的包）

如果 291 MB exe 太大，可以走 **Route B**（10 MB 源码 zip）：
- 我打一个 CoilTipViz-source.zip（已经实现 `scripts/gui/build_src_zip.py`）
- 你 Windows 装一次 anaconda + GPU torch (~5 min)
- 双击 `run_windows.bat` 启动
- 推理速度跟 exe 一样（甚至更快，源码无 PyInstaller overhead）

要 route B zip 我可以本地打给你。

---

## 不再依赖 GitHub Actions

把 `docs/windows_local_build_guide.md` commit 进 repo 后，`.github/workflows/build-exe.yml` 可以保留（作为参考），但本地跑是主路线。