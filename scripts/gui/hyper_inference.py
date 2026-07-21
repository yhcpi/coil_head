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

# ============================================================================
# ⚠️ CRITICAL: torch.load monkey-patch — 在 import ultralytics 之前必须执行
#
# 问题: ultralytics 8.0.227 nn/tasks.py:567 硬编码 `torch.load(file, map_location='cpu')`,
#       没传 weights_only. torch 2.5 默认 weights_only=False OK, 但 torch 2.6+ 默认改为
#       True → 加载 v26 best.pt (含 DetectionModel 等未注册类) 时抛 UnpicklingError:
#       "Unsupported global: GLOBAL ultralytics.nn.tasks.DetectionModel".
#       用户机器上 run_windows.bat 找到 Windows 的 Python, 可能是 torch 2.6+, 必报.
#
# 解决: 在最早期 monkey-patch torch.load 默认 weights_only=False. 这样:
#       - torch 2.5: 不变 (默认就是 False, 但 explicit 更稳)
#       - torch 2.6+: 强制 False, 跳过 weights_only 检查, ultralytics 加载 OK
#       - 安全: best.pt 是我们自己训练的, 来源可信, weights_only 安全检查没必要
#
# 此 patch 必须在 `from ultralytics import YOLO` 之前, 否则 ultralytics 已经 import
# 并缓存了原始 torch.load 引用 (ultralytics.nn.tasks.safe_load 调 torch.load).
# ============================================================================
def _apply_torch_load_weights_only_patch() -> None:
    """强制 torch.load 默认 weights_only=False, 兼容 torch 2.5/2.6/2.7."""
    try:
        import torch as _torch
    except ImportError:
        return  # torch 没装, 让后面 ImportError 自然抛
    _orig_load = _torch.load
    # 只 patch 一次 (避免循环)
    if getattr(_orig_load, "_weights_only_patched", False):
        return

    def _patched_load(*args, **kwargs):
        # 用户显式传了 weights_only → 尊重用户选择
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    _patched_load.__wrapped__ = _orig_load  # type: ignore[attr-defined]
    _patched_load._weights_only_patched = True  # type: ignore[attr-defined]
    _torch.load = _patched_load


_apply_torch_load_weights_only_patch()

# 项目根 = hyper_inference.py 的祖父目录 (scripts/gui/ -> scripts/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ----- 默认 SOTA 权重解析 -----
# 单一来源: v26 mid-strong full 300ep best.pt (F1=0.9359, 8MB)
# - 在 PyInstaller .exe 里: sys._MEIPASS/weights/best.pt
#   (build_exe.py 先 stage 权重到 _staged_best.pt, 再 --add-data "...;weights/" 放成 best.pt)
# - 在源码里: PROJECT_ROOT/runs/coil_panet_ablation/.../best.pt
SOURCE_DEPLOY_PT = (
    PROJECT_ROOT / "runs" / "coil_panet_ablation" / "v26_mid_strong_full_300ep" / "weights" / "best.pt"
)

# ----- PyInstaller bundle 支持 -----
def _find_bundle_weight(bundle_root: Path) -> Optional[Path]:
    """在 PyInstaller bundle 里找唯一的 SOTA 权重 (≥1MB)。

    build_exe.py 把 stage 后的 best.pt 放到 _internal/weights/best.pt。
    为了兼容不同 PyInstaller 版本，也 glob 整个 weights/ 目录里的 *.pt。
    """
    if not (bundle_root / "weights").exists():
        return None
    weights_dir = bundle_root / "weights"
    # 优先精确匹配
    exact = weights_dir / "best.pt"
    if exact.is_file() and exact.stat().st_size >= 1024 * 1024:
        return exact
    # fallback: 任何 ≥1MB .pt
    candidates = sorted(
        [p for p in weights_dir.glob("*.pt") if p.stat().st_size >= 1024 * 1024],
        key=lambda p: -p.stat().st_size,  # 最大的优先
    )
    return candidates[0] if candidates else None


def _resolve_default_deploy_pt() -> Optional[Path]:
    """返回第一个真实存在的 SOTA 权重 (≥1MB):
    1) PyInstaller bundle (.exe): sys._MEIPASS/weights/best.pt (或 glob)
    2) Route B zip (source-bundle, 无 frozen): module同目录 weights/best.pt
       - 适配 build_src_zip.py 路线, 解压 zip 后权重直接在同目录
    3) 源码 dev: PROJECT_ROOT/runs/.../best.pt
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundle_w = _find_bundle_weight(Path(sys._MEIPASS))
        if bundle_w:
            return bundle_w
    # Route B (zip 解压): weights/ 跟 hyper_inference.py 同目录
    sibling_w = Path(__file__).parent / "weights" / "best.pt"
    if sibling_w.is_file() and sibling_w.stat().st_size >= 1024 * 1024:
        return sibling_w
    if SOURCE_DEPLOY_PT.is_file() and SOURCE_DEPLOY_PT.stat().st_size >= 1024 * 1024:
        return SOURCE_DEPLOY_PT
    return None


DEFAULT_DEPLOY_PT = _resolve_default_deploy_pt()


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
    """解析最终要加载的模型权重路径。

    - 用户显式给了路径 → 必须存在且 ≥1MB；否则抛 FileNotFoundError
    - 缺省：用模块顶层算好的 DEFAULT_DEPLOY_PT (.exe bundle / 源码都支持)
      没找到 → 抛 FileNotFoundError 提示明确原因
    """
    if model_path:
        p = Path(model_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if _is_valid_weight_file(p):
            return p
        if p.is_file():
            size = p.stat().st_size
            raise FileNotFoundError(
                f"[HyperYoloDetector] 指定的权重文件存在但太小 (<1MB, 可能是 placeholder/损坏):\n"
                f"  路径: {p}\n"
                f"  大小: {size} bytes\n"
            )
        raise FileNotFoundError(
            f"[HyperYoloDetector] 指定的模型权重不存在:\n  路径: {p}\n"
        )
    # 缺省: 运行时重新解析 (不用模块顶部缓存, 因为 DEFAULT_DEPLOY_PT 在 import 时算, 文件可能后续变)
    runtime_pt = _resolve_default_deploy_pt()
    if runtime_pt is not None and _is_valid_weight_file(runtime_pt):
        return runtime_pt
    raise FileNotFoundError(
        "[HyperYoloDetector] 默认 SOTA 权重未找到 (>=1MB):\n"
        f"  - PyInstaller bundle: {Path(sys._MEIPASS) / 'weights' if hasattr(sys, '_MEIPASS') else 'N/A'}\n"
        f"  - 源码路径: {SOURCE_DEPLOY_PT}\n\n"
        "请手动选择 .pt 文件。"
    )


class HyperYoloDetector:
    """单类钢卷头/尾检测器 (默认加载部署 SOTA: v26 mid-strong full 300ep)。

    - 默认权重: v26 best.pt, F1=0.9359, 8MB
    - .exe 里: _internal/weights/best.pt
    - 源码里: runs/coil_panet_ablation/v26_mid_strong_full_300ep/weights/best.pt
    - 不使用 cfg= 参数 (兼容 ultralytics 8.0.227 / 8.3+)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,  # None → 自动用 DEFAULT_DEPLOY_PT
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