# Hyper-YOLO 环境配置指南

> **目标**：在你的 RTX 4060 Ti + Linux 上配通 Hyper-YOLO，能跑训练/推理
> **现状**：conda 环境 `hyper-yolo` (Python 3.12.13) 已建，pip 已配阿里云镜像，PyTorch 未装
> **关键约束**：本机 `github.com`/`pypi.org`/`download.pytorch.org` 443 被防火墙阻断，只能用阿里云镜像

---

## Step 0：激活环境（每次新终端都要做）

```bash
source /home/pi/anaconda3/etc/profile.d/conda.sh
conda activate hyper-yolo
```

**验证**：`python --version` 应该显示 `Python 3.12.13`

---

## Step 1：装 PyTorch（CUDA 12.4 + RTX 4060 Ti）

```bash
pip install torch torchvision torchaudio
```

**为什么用 pip 而不是 conda**：
- conda 走 nju 镜像，nju 的 pytorch channel 缺失（已验证）
- pip 已配阿里云镜像，PyTorch wheel 自带 CUDA runtime

**预期耗时**：3-10 分钟（包大，~2GB）

**验证**：
```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
**预期输出**（CUDA wheel 版本）：
```
torch: 2.5.x+cu124 (或 2.6.x+cu124)
cuda: True
device: NVIDIA GeForce RTX 4060 Ti
```

**如果失败**：
- 网络超时 → 重试 `pip install torch torchvision torchaudio`
- CUDA not available → 装的是 CPU 版 wheel，需要 `--index-url https://download.pytorch.org/whl/cu124`（但这域名被阻断，**改方案见 Step 1-备选**）

### Step 1-备选：CPU 版 PyTorch（最稳，但训练慢）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

如果 `download.pytorch.org` 也被阻断，**只能靠运气**——重试主命令多次。

---

## Step 2：装 Hyper-YOLO 的其他依赖

```bash
cd /home/pi/projects/hyperyolo/repos/Hyper-YOLO
pip install -r requirements.txt
```

**会装的包**：matplotlib, opencv-python, pillow, pyyaml, requests, scipy, tqdm, tensorboard, pandas, seaborn, psutil, thop, pycocotools

**预期耗时**：1-3 分钟

**验证**：
```bash
python -c "import cv2, numpy, torch, yaml, PIL, requests, scipy, tqdm; print('all good')"
```

**如果 pycocotools 失败**（Linux 上常见）：
```bash
pip install pycocotools -i https://mirrors.aliyun.com/pypi/simple/
# 或
pip install pycocotools-windows  # Windows 备选（Linux 用不上）
```

---

## Step 3：安装 Hyper-YOLO 本地包（editable 模式）

```bash
cd /home/pi/projects/hyperyolo/repos/Hyper-YOLO
pip install -e .
```

**为什么用 `-e`（editable）**：你会改 Hyper-YOLO 源码（按笔记加 Polar-C2Net/PA-Aug/Shape-NWD），editable 模式让代码改动立刻生效，不用重装。

**预期耗时**：30 秒

**验证**：
```bash
python -c "from ultralytics import YOLO; from ultralytics.nn.modules.block import *; print('Hyper-YOLO modules loaded')"
```

---

## Step 4：5 个 check 点（完整验证）

依次运行：

```bash
python << 'EOF'
import torch
print('1. torch version:', torch.__version__)
print('2. CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('   GPU:', torch.cuda.get_device_name(0))
    print('   CUDA:', torch.version.cuda)
    # 实际算一个小 tensor
    x = torch.randn(2, 3).cuda()
    y = (x + 1).sum()
    print('   GPU compute OK, sum:', y.item())

import ultralytics
print('3. ultralytics:', ultralytics.__version__)

from ultralytics import YOLO
print('4. YOLO class loaded')

# 5. 加载 Hyper-YOLO 配置（不下载权重，只读 yaml）
model = YOLO('ultralytics/cfg/models/hyper-yolo/hyper-yolon.yaml')
print('5. Hyper-YOLO-N config loaded, task:', model.task)
EOF
```

**预期全部通过**（这是关键 — 任何一步失败，下面 debug 章节有对应方案）

---

## Step 5：跑预训练权重 sanity check（可选，~100MB 下载）

⚠️ **这一步需要下权重文件**，如果网络慢可以跳过，等训练数据准备好再下。

```bash
cd /home/pi/projects/hyperyolo/repos/Hyper-YOLO

# 下预训练权重（COCO 训练好的 hyper-yolon.pt）
# 注意：可能下不动（github 阻断），失败就跳过
curl -L -o hyper-yolon.pt https://github.com/iMoonLab/Hyper-YOLO/releases/download/v1.0/hyper-yolon.pt

# 用 1.png 推理
python ultralytics/models/yolo/detect/predict.py \
  --weights hyper-yolon.pt \
  --source /home/pi/projects/hyperyolo/1.png \
  --img 640 \
  --save
```

**成功标志**：输出 `Results saved to runs/detect/predict/`

---

## Debug：常见错误及应对

### 错误 1：`No module named 'torch'` 
PyTorch 没装。回到 Step 1。

### 错误 2：`CUDA not available` 但 torch 是 +cu124
驱动问题。检查 `nvidia-smi` 是否能跑（应显示 RTX 4060 Ti）。

### 错误 3：`from ultralytics import YOLO` 报错 `ImportError`
可能装了新版 ultralytics 但 Hyper-YOLO 期望旧版。改用 `pip install -e .`（Step 3）。

### 错误 4：`pip install` 下载到一半 timeout
阿里云镜像偶尔抽风。重试 2-3 次，或者加 timeout：
```bash
pip install --timeout 600 torch torchvision torchaudio
```

### 错误 5：Hyper-YOLO 训练时 `ModuleNotFoundError: No module named 'pycocotools'`
回到 Step 2 末尾，手动装：
```bash
pip install pycocotools -i https://mirrors.aliyun.com/pypi/simple/
```

---

## 下一步（环境配通后）

按 `small_target_detection_v5_comprehensive.md` 的路线：

1. **W1**：准备数据集（标 100-200 张 1.png 同款 tip bbox）
   - 数据格式：YOLO txt（每张图一个 txt，每行 `class cx cy w h`，归一化）
   - 工具：LabelImg 或在线标注（Roboflow）
   - 目录结构：
     ```
     data/coil/
       images/{train,val}/*.jpg
       labels/{train,val}/*.txt
       data.yaml  # 见下面模板
     ```

2. **W2**：跑通 baseline 训练
   ```bash
   python ultralytics/models/yolo/detect/train.py \
     --model ultralytics/cfg/models/hyper-yolo/hyper-yolon.yaml \
     --data data/coil/data.yaml \
     --epochs 100 --imgsz 640 --batch 16 --device 0
   ```

3. **W3**：Stage 1 帧差触发 + Stage 2 端到端 demo

---

## data.yaml 模板（YOLO 格式）

```yaml
# data/coil/data.yaml
path: /home/pi/projects/hyperyolo/data/coil  # 绝对路径
train: images/train
val: images/val

# 类别数（钢卷头尾就 1 类）
nc: 1
names: ['coil_tip']
```

---

## 联系作者

- 环境装不上 → 把报错完整贴给我
- 网络奇葩 → 也贴（我之前测过：阿里云/USTC 通，pypi.org 不稳定）
- Hyper-YOLO 代码报错 → 看 Hyper-YOLO/ultralytics/ 源码或 GitHub issues