"""HyperYoloDetector — 轻量推理封装 for GUI / 部署脚本.

Usage:
    from scripts.gui.hyper_inference import HyperYoloDetector
    det = HyperYoloDetector()
    det.warmup((1024, 1024, 3))
    boxes = det.detect(frame)   # List[{'bbox': [x1,y1,x2,y2], 'conf': float, 'cls': int}]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# 项目根 = hyper_inference.py 的祖父目录 (scripts/gui/ -> scripts/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ----- PyInstaller bundle 支持 -----
# 在 .exe 运行时，weights 文件被 --add-data "weights;weights" 打到 sys._MEIPASS/weights/best.pt
# 但 PROJECT_ROOT 在 .exe 里指向用户解压目录（不是 _internal/），所以要走 sys._MEIPASS
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    BUNDLE_ROOT = Path(sys._MEIPASS)
    BUNDLE_WEIGHTS = BUNDLE_ROOT / "weights" / "best.pt"
else:
    BUNDLE_ROOT = None
    BUNDLE_WEIGHTS = None

# 默认权重候选 (按优先级)
#   在 .exe 里: sys._MEIPASS/weights/best.pt  (PyInstaller --add-data "weights;weights")
#   在源码里: PROJECT_ROOT/runs/coil_panet_ablation/v26_mid_strong_full_300ep/weights/best.pt
DEFAULT_DEPLOY_PT = (
    BUNDLE_WEIGHTS
    if BUNDLE_WEIGHTS is not None
    else PROJECT_ROOT / "runs" / "coil_panet_ablation" / "v26_mid_strong_full_300ep" / "weights" / "best.pt"
)
LEGACY_DEPLOY_PT = PROJECT_ROOT / "runs" / "deploy_best" / "v18_3_epoch60_hard_neg_weak_aug.pt"
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


def _is_valid_weight_file(p: Path) -> bool:
    """判断是否是真实的 .pt 权重（存在 + ≥1MB，排除 placeholder/空文件）。

    PyInstaller bundle 里如果 CI fallback 用 placeholder.pt，那个文件 0 字节，
    加载它会让 ultralytics 抛 'NoneType has no attribute encoding'。
    """
    if not p.is_file():
        return False
    try:
        return p.stat().st_size >= 1024 * 1024  # ≥1MB
    except OSError:
        return False


def _resolve_model_path(model_path: Optional[str]) -> Path:
    """缺省/不存在时自动 fallback 到历史 SOTA → hyper-yolon.pt。

    - 用户显式给的路径 → 校验必须存在且 ≥1MB，否则抛 FileNotFoundError（不静默 fallback）
    - 缺省：依次尝试 部署 SOTA (.exe:_MEIPASS/weights/best.pt / 源码:runs/...) → 历史 SOTA → fallback
    """
    if model_path:
        p = Path(model_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if _is_valid_weight_file(p):
            return p
        # 用户显式给了路径但找不到或太小 → 不静默 fallback
        if p.is_file():
            size = p.stat().st_size
            raise FileNotFoundError(
                f"[HyperYoloDetector] 指定的权重文件存在但太小 (<1MB, 可能是 placeholder/损坏):\n"
                f"  路径: {p}\n"
                f"  大小: {size} bytes\n\n"
                "请手动选一个真实的 v18.3 或 v26 .pt 文件。"
            )
        raise FileNotFoundError(
            f"[HyperYoloDetector] 指定的模型权重不存在:\n"
            f"  路径: {p}\n"
        )
    # 缺省: 部署 SOTA → 历史 SOTA → fallback
    for cand in (DEFAULT_DEPLOY_PT, LEGACY_DEPLOY_PT, FALLBACK_PT):
        if _is_valid_weight_file(cand):
            return cand
    raise FileNotFoundError(
        "[HyperYoloDetector] 自动 fallback 也未找到任何 ≥1MB 权重:\n"
        f"  - 部署 SOTA : {DEFAULT_DEPLOY_PT}\n"
        f"  - 历史 SOTA: {LEGACY_DEPLOY_PT}\n"
        f"  - fallback : {FALLBACK_PT}\n\n"
        "请手动选择 .pt 文件。"
    )


class HyperYoloDetector:
    """单类钢卷头/尾检测器 (默认加载部署 SOTA: v26 mid-strong full 300ep)。

    - 默认权重: runs/coil_panet_ablation/v26_mid_strong_full_300ep/weights/best.pt (F1=0.9359)
    - 历史 SOTA: runs/deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt (F1=0.9286)
    - 缺省/缺失 → 回退 repos/Hyper-YOLO/hyper-yolon.pt
    - 不使用 cfg= 参数 (兼容 8.0.227 / 8.3+)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,  # None → 自动 fallback 链 (部署 SOTA → 历史 SOTA → base)
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