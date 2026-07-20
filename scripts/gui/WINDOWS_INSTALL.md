# 钢卷头尾检测 GUI — Windows 用户手册

## 🎯 一句话说明

双击桌面 **"钢卷头尾检测 GUI"** 快捷方式 → 选权重文件 → 选视频 → 点 ▶ 开始。

---

## 📦 安装步骤 (只需一次)

### 方案 A：用打包好的 .exe (无需 Python 环境)

适合：车间工控机、客户电脑、不想装 Python 的用户。

1. 拿到 `CoilTipViz.zip` 压缩包
2. 解压到任意目录 (例如 `D:\CoilTipViz\`)
3. 双击 `D:\CoilTipViz\CoilTipViz.exe` 启动
4. (可选) 右键 → 发送到 → 桌面快捷方式

### 方案 B：从源码启动 (需要 Python 环境)

适合：开发人员、需要改代码的用户。

#### 1. 安装 Anaconda

下载: <https://www.anaconda.com/download>

#### 2. 创建 hyper-yolo 环境

打开 **Anaconda Prompt**:

```bat
conda create -n hyper-yolo python=3.10 -y
conda activate hyper-yolo
pip install ultralytics==8.0.227 opencv-python pillow pyav
```

#### 3. 下载项目代码

把整个 `hyperyolo` 项目文件夹放到本地, 例如:

```
C:\projects\hyperyolo\
  └─ scripts\gui\
       ├─ coil_tip_viz_gui.py    ← 主程序
       ├─ frame_diff_wrapper.py
       └─ hyper_inference.py
```

#### 4. 准备帧差法代码 (可选, 不装也能用)

把 `pi\projects\mm\帧差法` 文件夹放到:

```
C:\projects\mm\帧差法\
```

> ⚠ 如果不放, GUI 会跳过帧差法功能 (不截图, 仅显示视频 + bbox)。

#### 5. 创建桌面快捷方式

双击 `C:\projects\hyperyolo\scripts\gui\create_shortcut.bat`

桌面会出现 **"钢卷头尾检测 GUI"** 图标。

---

## 🚀 使用流程

### 第一次启动

1. **双击桌面快捷方式** → 弹出 GUI 窗口
2. 程序自动加载默认权重 (v18.3 部署模型)
3. 如果加载失败, 点 **"选择权重 (.pt)"** → 选你的 .pt 文件

### 处理视频

1. 点 **"选择视频文件"** → 选一个或多个 .mp4 / .avi
2. (可选) 点 **"选择输出目录"** → 选截图保存位置 (默认 `runs\管控\captures_gui\`)
3. 点 **"▶ 开始"** → 自动处理

### 界面说明

```
┌────────────────────────────────────────────────────────────┐
│ [选择权重] [选择视频文件] [选择文件夹] [选择输出目录] [▶ 开始] │ ← 工具栏
│ 权重: v18_3_*.pt | 输出: ... | 视频: 1.MP4                    │
├──────────┬───────────────────────────────────┬──────────────┤
│ 视频列表 │         当前帧 + bbox              │ 状态        │
│          │         (钢卷 tip 绿框)           │ STABLE      │
│ 1.MP4    │                                   │ CHANGE_RIS..│
│ 2.MP4    │                                   │ FPS: 12.3   │
│ 3.MP4    │                                   ├──────────────┤
│          │                                   │ 检测结果    │
│          │                                   │ conf=0.87   │
│          │                                   ├──────────────┤
│          │                                   │ 截图列表    │
│          │                                   │ change_01.. │
├──────────┴───────────────────────────────────┴──────────────┤
│ [上一段]  [下一段]   段: 1/5                                  │
└────────────────────────────────────────────────────────────┘
```

---

## 🔧 高级

### 打包自己的 .exe

```bat
cd C:\projects\hyperyolo\scripts\gui
build_exe.bat
```

输出: `dist\CoilTipViz\CoilTipViz.exe` + 依赖目录 (~1-2GB)

### 切换模型

GUI 里点 **"选择权重"** 可加载任意 .pt 文件:

- `runs\deploy_best\v18_3_*.pt` (F1=0.9286, 部署最优)
- `runs\deploy_best\v19_*\best.pt` (学术 SOTA)
- 任意训练的 best.pt / last.pt

### 切换置信度阈值

编辑 `coil_tip_viz_gui.py` 顶部:

```python
CONF_HI = 0.5  # bbox 颜色分界 (绿/黄)
```

或在 `hyper_inference.py` 改 `conf` 默认值。

### 每秒截图数量

编辑 `coil_tip_viz_gui.py` 中:

```python
fd_proc = FrameDiffProcessor(
    out_dir=output_dir,
    capture_times_sec=[0.0, 1.0],  # 改成 [0.0, 1.0, 2.0, 3.0] 截 4 张
)
```

---

## ❓ 故障排查

| 症状 | 原因 | 修复 |
|---|---|---|
| 双击 exe 闪退 | 缺 VC++ 运行库 | 装 [VC++ Redist](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| "ModuleNotFoundError: No module named 'av'" | 漏装 pyav | `pip install pyav` |
| "模型加载失败" | 权重路径错 | 点 "选择权重" 重选 |
| 视频打不开 | 格式不支持 | 转 .mp4 (H.264) 重试 |
| "帧差法初始化失败" | 帧差法目录不存在 | 装 `C:\projects\mm\帧差法` 或忽略 (帧差法可选) |
| GPU 没用到 | CUDA 没装 | `pip install torch==2.5.1+cu124` |

---

## 📂 输出位置

- **截图**: `<输出目录>\<视频名>\change_XX_*.png`
- **默认**: `C:\projects\hyperyolo\runs\captures_gui\<视频名>\`

每段 change 段保存:
- `change_01_RISING_0.50s.png` (起始瞬间)
- `change_01_RISING_1.50s.png` (持续 1 秒)
- `change_02_FALLING_0.30s.png` (下一段开始)

---

## 🆚 三种使用方式对比

| 方式 | 优点 | 缺点 | 适合 |
|---|---|---|---|
| 双击 .exe (方案 A) | 无需 Python, 直接用 | 包大 (~1.5GB), 启动慢 | 工控机/客户 |
| 双击 .bat (方案 B) | 包小, 启动快 | 需 Python 环境 | 开发者 |
| 命令行启动 | 可调试 | 不直观 | 高级用户 |

---

## 联系

遇到问题截图发回项目仓库 issue。