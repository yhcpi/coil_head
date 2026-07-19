# 训练配置说明（v10_nbbox_light_smoke）

- 保存时间：2026-07-11 12:54:09
- 训练目录：runs/v10_nbbox_light/v10_nbbox_light_smoke
- 启动用户：pi
- 主机：BF-202412171403
- PyTorch：2.5.1+cu124
- CUDA available：True
- GPU：NVIDIA GeForce RTX 4060 Ti

## 训练命令

```bash
python -m ultralytics.models.yolo.detect.train task=detect mode=train model=hyper-yolon.yaml data=data/coil/data.yaml epochs=10 patience=0 batch=16 imgsz=1024 save=False val_period=2 start_val_epoch=2 save_period=-1 cache=False device=0 workers=0 project=runs/v10_nbbox_light name=v10_nbbox_light_smoke exist_ok=True pretrained=False optimizer=SGD verbose=True seed=0 deterministic=True single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True fraction=1.0 profile=False overlap_mask=True mask_ratio=4 dropout=0.0 val=True split=val save_json=False save_hybrid=False conf=0.001 iou=0.7 max_det=300 half=True dnn=False plots=True vid_stride=1 stream_buffer=False visualize=False augment=False agnostic_nms=False retina_masks=False show=False save_frames=False save_txt=False save_conf=False save_crop=False show_labels=False show_conf=False show_boxes=False line_width=1 format=onnx keras=False optimize=False int8=False dynamic=False simplify=True workspace=4 nms=False lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 box=1.5 cls=0.5 dfl=1.5 nwd=True nwd_constant=12.0 gwd=False gwd_tau=1.0 pose=12.0 kobj=1.0 label_smoothing=0.0 coverage=False coverage_weight=0.5 coverage_sigma=20.0 looseness_alpha=0.0 looseness_target_area=400.0 stal_area_thr=0.0 stal_topk=13 stal_expand=0.2 bbox_shrink_min=1.0 bbox_shrink_max=1.0 bbox_shrink_p=0.0 bbox_noise=True bbox_noise_scale=0.8,1.2 bbox_noise_shift=0.05 bbox_noise_p=0.3 spec_suppress=False spec_recon_weight=0.1 multi_scale=0.0 nbs=64 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 degrees=0.0 translate=0.0 scale=0.0 shear=0.0 perspective=0.0 flipud=0.0 fliplr=0.5 mosaic=0.0 mixup=0.0 copy_paste=0.0 tracker=botsort.yaml save_dir=runs/v10_nbbox_light/v10_nbbox_light_smoke
```

## 训练参数

| 参数 | 值 |
|---|---|
| epochs | 10 |
| batch | 16 |
| imgsz | 1024 |
| device | 0 |
| workers | 0 |
| rect | True |
| freeze | None |
| pretrained | False |
| patience | 0 |
| optimizer | SGD |
| project | runs/v10_nbbox_light |
| name | v10_nbbox_light_smoke |

## 数据集状态

- yaml: data/coil/data.yaml
- train_images: 545
- train_labels_total: 545
- train_labels_with_target: 312
- train_labels_empty (负样本): 233
- val_images: 99
- val_labels_total: 99
- val_labels_with_target: 43
- val_labels_empty (负样本): 56

## 创新点 / 改动清单

1. **数据策略**：移 27 张 val-only FN 到 train（val 减到 102 张）
2. **bbox_random_shrink**：每个 GT bbox 边长随机缩放 0.8~1.2 倍，保持中心
3. **Coverage Loss**：pred bbox 包住 GT 中心的概率作为附加 loss 项，权重 0.5
4. **multi-scale rect**：imgsz ±20% 随机，保持长宽比不变
5. **loss 权重微调**：box 7.5→5.0（降），cls 0.5→1.0（升），label_smoothing 0.05→0.02（降）
6. **copy_paste**：0.1 → 0.2
7. **从预训练权重开始**（不加载 last.pt）

## hyp_aug.yaml（实际生效版本）

```yaml

```

## 模型配置

- model yaml: `hyper-yolon.yaml`

## 源码改动

- Coverage Loss 实现：`ultralytics/utils/loss.py (coverage_loss 函数)`
- bbox_random_shrink + multi-scale 实现：`ultralytics/data/augment.py (BBoxRandomShrink + RandomScaleRect)`
