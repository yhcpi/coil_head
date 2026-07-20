# 钢卷头尾检测可视化软件 (Coil Tip Detection Visualization)

## 功能
- 选择本地视频文件夹, 自动扫描视频文件 (.mp4/.avi/.mov)
- 实时帧差法 change 状态检测 (集成 /home/pi/projects/mm/帧差法)
- change 状态每秒截图一张
- 同步显示 hyper-yolo 钢卷线头检测结果 (bbox + conf)

## 安装
环境: /home/pi/anaconda3/envs/hyper-yolo (已装 ultralytics 8.0.227)
依赖: tkinter (标准库) + PIL + opencv-python + ultralytics + pyav

## 启动
```bash
cd /home/pi/projects/hyperyolo
PYTHONPATH=/home/pi/projects/mm/帧差法 /home/pi/anaconda3/envs/hyper-yolo/bin/python scripts/gui/coil_tip_viz_gui.py
```

## 使用
1. 点击 "选择文件夹" 选视频目录
2. 左侧列表选视频
3. 点击 "开始" 自动处理
4. 中央显示当前帧 (含 bbox 叠加)
5. 右侧显示 change 状态 + 检测结果 + 截图列表

## 文件结构
- coil_tip_viz_gui.py: 主 GUI
- frame_diff_wrapper.py: 帧差法包装
- hyper_inference.py: hyper-yolo 推理包装
- 截图保存: runs/captures_gui/{video_name}/

## 模型
- 部署权重: runs/deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt (F1=0.9286)
- fallback: repos/Hyper-YOLO/hyper-yolon.pt

## 配置参数 (coil_tip_viz_gui.py 顶部)
- CONF_THRESHOLD = 0.15 (hyper-yolo 检测阈值)
- CAPTURE_TIMES_SEC = [0.0, 1.0] (每秒截图)

## 已知限制
- WSL2 环境下 GUI 显示需 X server (Xming/VcXsrv) 或 WSLg
- 单视频串行处理, 不支持多视频并行
- 模型推理用 GPU (device='0'), 没 GPU 会自动 fallback CPU

## 故障排查
- ImportError: No module named 'frame_diff_detector' -> 检查 PYTHONPATH
- 模型加载失败 -> 检查 runs/deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt 是否存在
- 视频读不了 -> 检查 pyav/cv2 安装

## 后续改进
- 多视频并行处理
- 实时导出 change 段视频
- 检测结果数据库持久化