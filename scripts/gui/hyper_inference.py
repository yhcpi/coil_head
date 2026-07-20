"""HyperYoloDetector — 轻量推理封装 for GUI / 部署脚本.

Usage:
    from scripts.gui.hyper_inference import HyperYoloDetector
    det = HyperYoloDetector()
    det.warmup((1024, 1024, 3))
    boxes = det.detect(frame)   # List[{'bbox': [x1,y1,x2,y2], 'conf': float, 'cls': int}]
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# 项目根 = hyper_inference.py 的祖父目录 (scripts/gui/ -> scripts/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 默认权重候选 (按优先级)
DEFAULT_DEPLOY_PT = PROJECT_ROOT / "runs" / "deploy_best" / "v18_3_epoch60_hard_neg_weak_aug.pt"
FALLBACK_PT = PROJECT_ROOT / "repos" / "Hyper-YOLO" / "hyper-yolon.pt"


def _resolve_device(device: Union[str, int]) -> str:
    """有 CUDA 走 '0'，否则 'cpu'。接受 'auto' / 'cuda' / 索引 / 'cpu'。"""
    if isinstance(device, str) and device not in ("auto", "", None):
        s = device.strip().lower()
        if s == "cpu":
            return "cpu"
        if s in ("0", "1", "2", "3", "cuda", "cuda:0", "gpu"):
            try:
                import torch  # noqa: WPS433
                if torch.cuda.is_available():
                    return "0" if s in ("cuda", "gpu", "cuda:0") else s
            except ImportError:
                pass
            return "cpu"
    try:
        import torch  # noqa: WPS433
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _resolve_model_path(model_path: Optional[str]) -> Path:
    """缺省/不存在时自动 fallback 到 hyper-yolon.pt。"""
    if model_path:
        p = Path(model_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file():
            return p
        # 用户显式给了路径但找不到 → 不静默 fallback，给清晰错误
        raise FileNotFoundError(
            f"[HyperYoloDetector] 指定的模型权重不存在: {p}\n"
            f"  部署权重候选: {DEFAULT_DEPLOY_PT}\n"
            f"  fallback 候选: {FALLBACK_PT}"
        )
    # 缺省: 部署权重 → fallback
    if DEFAULT_DEPLOY_PT.is_file():
        return DEFAULT_DEPLOY_PT
    if FALLBACK_PT.is_file():
        return FALLBACK_PT
    raise FileNotFoundError(
        "[HyperYoloDetector] 未找到任何可用权重:\n"
        f"  - 部署权重: {DEFAULT_DEPLOY_PT}\n"
        f"  - fallback : {FALLBACK_PT}\n"
        "请先训练或下载 hyper-yolon.pt。"
    )


class HyperYoloDetector:
    """单类钢卷头/尾检测器 (Hyper-YOLON + NWD v18.3 部署权重)。

    - 默认权重: runs/deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt
    - 缺省/缺失 → 回退 repos/Hyper-YOLO/hyper-yolon.pt
    - 不使用 cfg= 参数 (兼容 8.0.227 / 8.3+)
    """

    def __init__(
        self,
        model_path: str = "runs/deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt",
        conf: float = 0.15,
        imgsz: int = 1024,
        device: str = "0",
    ) -> None:
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = _resolve_device(device)
        self.model_path = _resolve_model_path(model_path)

        try:
            from ultralytics import YOLO  # noqa: WPS433
        except ImportError as exc:
            raise RuntimeError(
                "[HyperYoloDetector] 未安装 ultralytics。\n"
                "  请在 hyper-yolo 环境下运行:\n"
                "  /home/pi/anaconda3/envs/hyper-yolo/bin/python ..."
            ) from exc

        try:
            self.model = YOLO(str(self.model_path))
            # ultralytics.predict() 接受 '0' / 'cpu', 但 nn.Module.to() 要 cuda:0
            torch_device = "cuda:0" if self.device not in ("cpu", "mps") else self.device
            self.model.to(torch_device)
        except Exception as exc:
            raise RuntimeError(
                f"[HyperYoloDetector] 加载模型失败: {self.model_path}\n"
                f"  device={self.device}  imgsz={self.imgsz}\n"
                f"  原错误: {type(exc).__name__}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    def warmup(self, frame_shape: Tuple[int, int, int] = (1024, 1024, 3)) -> None:
        """用一张 dummy frame 预热，避免首次推理卡顿。"""
        h, w = int(frame_shape[0]), int(frame_shape[1])
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        try:
            self.detect(dummy)
        except Exception:
            # warmup 失败不应阻断 detector 本身的使用
            pass

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """单帧推理，返回 [{bbox, conf, cls}, ...] 列表。"""
        if frame is None:
            return []
        try:
            results = self.model.predict(
                frame,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[HyperYoloDetector] predict 失败: {type(exc).__name__}: {exc}"
            ) from exc

        out: List[Dict[str, Any]] = []
        if not results:
            return out
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.zeros((0, 4))
        confs = (
            boxes.conf.cpu().numpy()
            if hasattr(boxes, "conf") and boxes.conf is not None
            else np.zeros((len(xyxy),), dtype=np.float32)
        )
        clses = (
            boxes.cls.cpu().numpy().astype(int)
            if hasattr(boxes, "cls") and boxes.cls is not None
            else np.zeros((len(xyxy),), dtype=int)
        )

        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clses):
            out.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(c),
                    "cls": int(k),
                }
            )
        return out


# ----------------------------------------------------------------------
# CLI smoke: python -m scripts.gui.hyper_inference <image_path>
if __name__ == "__main__":  # pragma: no cover
    import sys

    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    det = HyperYoloDetector()
    det.warmup()
    if img_path and os.path.isfile(img_path):
        import cv2

        img = cv2.imread(img_path)
        boxes = det.detect(img)
        print(f"[smoke] {os.path.basename(img_path)} -> {len(boxes)} det")
        for b in boxes[:5]:
            print(f"  bbox={b['bbox']}  conf={b['conf']:.3f}  cls={b['cls']}")
    else:
        print(f"[smoke] detector ready, weight={det.model_path}, device={det.device}")