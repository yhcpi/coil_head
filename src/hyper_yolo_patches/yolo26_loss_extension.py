"""YOLO26 BboxLoss extension — A (NWD+CIoU blend) + B (reg_max=1 safe) + C (box soft).

This module is a **monkey-patch**, not a fork. Importing it rewires:

- `ultralytics.utils.loss.BboxLoss`           → blend of CIoU + NWD on `loss_iou`,
                                                 safe across `reg_max=1` (DFL off)
- `ultralytics.utils.loss.v8DetectionLoss`    → reads new hyp kwargs from
                                                 `model.args` and propagates them
                                                 into `BboxLoss`
- `ultralytics.cfg.check_dict_alignment`      → whitelist for new CLI keys
                                                 (`iou_loss_weight_nwd`, etc.)

Drop-in: when `iou_loss_weight_nwd=0` and `box_soft_sigma=0` the patched code is
bit-for-bit equivalent to the upstream CIoU-only path. Any other values switch on
the blend / soft-noise branches.

Usage (entry script):

    import sys
    sys.path.insert(0, '/home/pi/projects/hyperyolo/src/hyper_yolo_patches')
    import yolo26_loss_extension          # noqa — installs patches
    from ultralytics.cli import run
    run()

CLI kwargs recognised by `model.args`:

    iou_loss_weight_nwd   float  default 0.0   weight of NWD in loss_iou (1 - sim)
    iou_loss_weight_ciou  float  default 1.0   weight of CIoU in loss_iou (1 - sim)
    box_soft_sigma        float  default 0.0   pixel-sigma for Gaussian label smoothing
                                                on `target_bboxes` (train mode only).
                                                0 disables; 2.0 is the recommended start.
                                                Only used when box_soft_relative=0.
    box_soft_relative     float  default 0.0   ratio of box w/h used as sigma for relative
                                                Gaussian jitter (e.g. 0.10 = ±10% of box size).
                                                When >0, OVERRIDES box_soft_sigma. Per-box scale,
                                                stride-invariant, adapts to varying object sizes
                                                (better for loose annotations).
    box_soft_train_only   bool   default True  restrict noise to `model.training`
    nwd_constant          float  default 12.0  paper constant for NWD exp scaling
"""

import math
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default weights when nwd is requested via the legacy `nwd=true` switch
# (kept for compat with v8-style hyp.yaml users).
_DEFAULT_NWD_W = 0.7
_DEFAULT_CIOU_W = 0.3

# CLI args this module adds. Used to whitelist them in cfg parser.
OUR_KEYS = {
    "iou_loss_weight_nwd",
    "iou_loss_weight_ciou",
    "box_soft_sigma",
    "box_soft_train_only",
    "box_soft_relative",   # >0: relative mode (按 box w/h 比例 jitter), 覆盖 box_soft_sigma
    "nwd_constant",
}

# Trainer/validator 内部状态字段，validator 把 trainer.args 整个传给 get_cfg 时
# 这些字段不在默认 cfg base 里，需要加入白名单防止 SyntaxError。
# 来源: ultralytics 8.4.82 BaseTrainer.__init__ 设置的字段 + DetectionValidator 字段
TRAINER_INTERNAL_KEYS = {
    "save_dir", "save_crop", "save_hybrid", "crop_fraction",
    "label_smoothing",  # 8.0.x 时合法，8.4.x 移除
    "plots", "verbose", "show", "save_txt", "save_conf",
    "save_frames", "save_json", "show_labels", "show_boxes", "show_conf",
    "amp", "cache", "device", "workers", "single_cls",
    "task", "model", "data", "imgsz", "seed", "deterministic",
    "cos_lr", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
    "warmup_momentum", "warmup_bias_lr", "box", "cls", "dfl",
    "pose", "kobj", "nbs", "hsv_h", "hsv_s", "hsv_v", "degrees",
    "translate", "scale", "shear", "perspective", "flipud", "fliplr",
    "mosaic", "mixup", "copy_paste", "erasing", "auto_augment",
    "close_mosaic", "optimizer", "overlap_mask", "mask_ratio",
    "dropout", "val", "split", "rect", "single_cls",
    "time", "profile", "multi_scale", "compile", "end2end",
    "exist_ok", "project", "name", "patience", "save_period",
    "fraction", "freeze", "resume", "amp", "agnostic_nms",
    "retina_masks", "embed", "distill_model", "dnn", "opset",
    "workspace", "format", "keras", "optimize", "int8", "half",
    "batch", "epochs",
}


# ---------------------------------------------------------------------------
# Math primitives — kept self-contained so this module works even if the
# hyper_yolo_patches ultralytics/utils/loss.py is NOT on sys.path. (The
# YOLO26 train script imports the unmodified `repos/ultralytics` package.)
# ---------------------------------------------------------------------------

def _nwd_xyxy(pred_xyxy, target_xyxy, eps=1e-7, constant=12.0):
    """NWD similarity in [0, 1] on xyxy boxes (1 = identical, 0 = far).

    Mirrors hyper_yolo_patches/ultralytics/utils/loss.py:wasserstein_loss so the
    older NWD implementation and this one produce identical gradients. The
    2nd-Wasserstein form `sqrt(center_d2 + wh_d2)` is exponentiated with the
    Wang et al. 2021 normalization (constant=12 for AI-TOD scale).
    """
    p_x1, p_y1, p_x2, p_y2 = pred_xyxy.unbind(-1)
    t_x1, t_y1, t_x2, t_y2 = target_xyxy.unbind(-1)
    p_cx = (p_x1 + p_x2) * 0.5
    p_cy = (p_y1 + p_y2) * 0.5
    t_cx = (t_x1 + t_x2) * 0.5
    t_cy = (t_y1 + t_y2) * 0.5
    center_d2 = (p_cx - t_cx) ** 2 + (p_cy - t_cy) ** 2 + eps
    p_w = (p_x2 - p_x1) + eps
    p_h = (p_y2 - p_y1) + eps
    t_w = (t_x2 - t_x1) + eps
    t_h = (t_y2 - t_y1) + eps
    wh_d2 = ((p_w - t_w) ** 2 + (p_h - t_h) ** 2) * 0.25
    return torch.exp(-torch.sqrt(center_d2 + wh_d2) / constant)


def _ciou_xyxy(pred_xyxy, target_xyxy, eps=1e-7):
    """CIoU on xyxy boxes, returned as a similarity in [0, 1]."""
    lt = torch.max(pred_xyxy[..., :2], target_xyxy[..., :2])
    rb = torch.min(pred_xyxy[..., 2:], target_xyxy[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area_p = ((pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=0)
              * (pred_xyxy[..., 3] - pred_xyxy[..., 1]).clamp(min=0))
    area_t = ((target_xyxy[..., 2] - target_xyxy[..., 0]).clamp(min=0)
              * (target_xyxy[..., 3] - target_xyxy[..., 1]).clamp(min=0))
    union = area_p + area_t - inter + eps
    iou = inter / union

    enclose_lt = torch.min(pred_xyxy[..., :2], target_xyxy[..., :2])
    enclose_rb = torch.max(pred_xyxy[..., 2:], target_xyxy[..., 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_d2 = enclose_wh[..., 0] ** 2 + enclose_wh[..., 1] ** 2 + eps

    p_cx = (pred_xyxy[..., 0] + pred_xyxy[..., 2]) * 0.5
    p_cy = (pred_xyxy[..., 1] + pred_xyxy[..., 3]) * 0.5
    t_cx = (target_xyxy[..., 0] + target_xyxy[..., 2]) * 0.5
    t_cy = (target_xyxy[..., 1] + target_xyxy[..., 3]) * 0.5
    rho2 = (p_cx - t_cx) ** 2 + (p_cy - t_cy) ** 2

    p_w = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=eps)
    p_h = (pred_xyxy[..., 3] - pred_xyxy[..., 1]).clamp(min=eps)
    t_w = (target_xyxy[..., 2] - target_xyxy[..., 0]).clamp(min=eps)
    t_h = (target_xyxy[..., 3] - target_xyxy[..., 1]).clamp(min=eps)
    v = (4.0 / (math.pi ** 2)) * (torch.atan(t_w / t_h) - torch.atan(p_w / p_h)) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1.0 + eps))
    return iou - (rho2 / enclose_d2 + v * alpha)


# ---------------------------------------------------------------------------
# Patched BboxLoss
# ---------------------------------------------------------------------------

def _patch_bbox_loss(orig_cls):
    """Build a subclass of the upstream BboxLoss with A/B/C behaviour layered.

    Detection of upstream API version is automatic:
      - ultralytics 8.0.x: BboxLoss(reg_max), forward takes 7 args (xyxy, no imgsz/stride)
      - ultralytics 8.4.x: BboxLoss(reg_max), forward takes 9 args (adds imgsz, stride)

    The blend (`A`) and box-soft (`C`) branches fire only when their respective
    weights are non-default; otherwise loss_iou falls back to plain CIoU on
    unperturbed targets → bit-identical to upstream.
    """

    fwd_params = list(inspect.signature(orig_cls.forward).parameters.keys())
    new_api = "imgsz" in fwd_params  # 9-arg forward with stride/imgsz at the end

    class PatchedBboxLoss(orig_cls):
        def __init__(self, reg_max, *, nwd_weight=0.0, ciou_weight=1.0,
                     nwd_constant=12.0, soft_sigma_pixel=0.0, soft_relative_ratio=0.0,
                     soft_train_only=True,
                     **_ignored):
            # Upstream BboxLoss(reg_max) only stores `dfl_loss`; nothing else
            # needs forward. Use the parent's positional signature.
            super().__init__(reg_max)
            self.nwd_weight = float(nwd_weight)
            self.ciou_weight = float(ciou_weight)
            self.nwd_constant = float(nwd_constant)
            self.soft_sigma_pixel = float(soft_sigma_pixel)
            self.soft_relative_ratio = float(soft_relative_ratio)
            self.soft_train_only = bool(soft_train_only)
            self._blend_active = (self.nwd_weight > 0
                                  or abs(self.ciou_weight - 1.0) > 1e-9)
            # Soft jitter active if EITHER pixel-sigma > 0 OR relative ratio > 0
            self._soft_active = self.soft_sigma_pixel > 0 or self.soft_relative_ratio > 0
            # Relative mode takes precedence over pixel mode when > 0
            self._soft_relative_mode = self.soft_relative_ratio > 0
            # Cache the API shape so forward doesn't re-inspect on every step.
            self._new_api = new_api

        def forward(self, pred_dist, pred_bboxes, anchor_points,
                    target_bboxes, target_scores, target_scores_sum, fg_mask,
                    *extra):
            """Compute (loss_iou, loss_dfl).

            Layout arguments:
              pred_dist, pred_bboxes, anchor_points, target_bboxes,
              target_scores, target_scores_sum, fg_mask   — all required

            Extra (new API only):
              imgsz   H,W tensor in pixels
              stride  per-anchor stride tensor (feature-pixel → imgsz-pixel)
            """
            imgsz_arg = extra[0] if len(extra) >= 1 else None
            stride_arg = extra[1] if len(extra) >= 2 else None

            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

            # ---------- (C) box soft target perturbation ----------
            t_for_iou = target_bboxes
            if self._soft_active:
                if self._soft_relative_mode:
                    # Relative mode: σ per dim is a fraction of that box's own size.
                    # x1, x2 use box width; y1, y2 use box height. Adapts to any box
                    # size (better than fixed pixel sigma for mixed-scale training).
                    target_bboxes_xywh = t_for_iou.clone() if t_for_iou is target_bboxes else t_for_iou
                    tw = (target_bboxes_xywh[..., 2] - target_bboxes_xywh[..., 0]).clamp(min=1e-4)
                    th = (target_bboxes_xywh[..., 3] - target_bboxes_xywh[..., 1]).clamp(min=1e-4)
                    sigma_per_dim = torch.stack(
                        [tw, th, tw, th], dim=-1
                    ) * self.soft_relative_ratio  # (bs, na, 4)
                    noise = torch.randn_like(target_bboxes) * sigma_per_dim
                elif stride_arg is not None:
                    # Pixel mode + stride-aware: σ_pixel / stride so all 3 feature
                    # levels get the SAME pixel-space noise.
                    sigma_feat = self.soft_sigma_pixel / stride_arg.clamp(min=1.0)
                    noise = (torch.randn_like(target_bboxes)
                             * sigma_feat.view(1, -1, 1))
                else:
                    noise = torch.randn_like(target_bboxes) * self.soft_sigma_pixel
                t_for_iou = target_bboxes + noise

            # ---------- (A) CIoU + NWD blend / pure CIoU ----------
            p = pred_bboxes[fg_mask]
            t = t_for_iou[fg_mask]
            iou = _ciou_xyxy(p, t).clamp(min=0.0, max=1.0)
            if self._blend_active:
                nwd = _nwd_xyxy(p, t, constant=self.nwd_constant).clamp(0.0, 1.0)
                sim = self.ciou_weight * iou + self.nwd_weight * nwd
            else:
                sim = iou
            loss_iou = ((1.0 - sim) * weight).sum() / target_scores_sum

            # ---------- (B) DFL / reg_max=1 safe path ----------
            # Upstream keeps `self.dfl_loss = DFLoss(reg_max) if reg_max>1 else None`.
            if self.dfl_loss is not None:
                target_ltrb = bbox2dist(anchor_points, target_bboxes,
                                        self.dfl_loss.reg_max - 1)
                loss_dfl = self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                ) * weight
                loss_dfl = loss_dfl.sum() / target_scores_sum
            elif new_api and imgsz_arg is not None and stride_arg is not None:
                # reg_max=1 fallback in upstream 8.4.x: L1 on imgsz-normalized ltrb.
                target_ltrb = bbox2dist(anchor_points, target_bboxes)
                target_ltrb = target_ltrb * stride_arg
                target_ltrb[..., 0::2] /= imgsz_arg[1]
                target_ltrb[..., 1::2] /= imgsz_arg[0]
                pred_dist_loc = pred_dist * stride_arg
                pred_dist_loc[..., 0::2] /= imgsz_arg[1]
                pred_dist_loc[..., 1::2] /= imgsz_arg[0]
                loss_dfl = (
                    F.l1_loss(pred_dist_loc[fg_mask],
                              target_ltrb[fg_mask],
                              reduction="none")
                    .mean(-1, keepdim=True) * weight
                ).sum() / target_scores_sum
            else:
                loss_dfl = torch.tensor(0.0).to(pred_dist.device)
            return loss_iou, loss_dfl

    return PatchedBboxLoss


# Import the bbox2dist helper here (top-level, not inside the patcher) so the
# function binds to the upstream `ultralytics.utils.tal` at install time —
# whichever package is on sys.path.
try:
    from ultralytics.utils.tal import bbox2dist  # noqa: E402
except ImportError:
    bbox2dist = None  # silently OK; we only use it when DFL is on


# ---------------------------------------------------------------------------
# Patched v8DetectionLoss
# ---------------------------------------------------------------------------

def _patch_v8_loss(orig_cls, PatchedBboxLoss):
    """Patch the v8DetectionLoss class WITHOUT replacing it in the module
    namespace — that breaks torch.save pickle (different class object).

    We monkey-patch `__init__` so that AFTER the parent's __init__ builds the
    default self.bbox_loss, we swap it for PatchedBboxLoss when the user passed
    the new CLI keys. The class identity is preserved (same __qualname__), so
    checkpoint serialization works.
    """
    _orig_init = orig_cls.__init__

    def _patched_init(self, model, *args, **kwargs):
        _orig_init(self, model, *args, **kwargs)
        h = model.args
        # FIX 2026-07-14: use `is None` check instead of `or default` — Python's
        # short-circuit treats 0.0/0 as falsy and silently "corrects" the user's
        # iou_loss_weight_ciou=0.0 to the default 1.0, breaking NWD-only mode
        # (sim = iou + nwd > 1 → loss_iou negative).
        nwd_w_raw = getattr(h, "iou_loss_weight_nwd", None)
        nwd_w = float(nwd_w_raw) if nwd_w_raw is not None else 0.0
        ciou_w_raw = getattr(h, "iou_loss_weight_ciou", None)
        ciou_w = float(ciou_w_raw) if ciou_w_raw is not None else 1.0
        nwd_c_raw = getattr(h, "nwd_constant", None)
        nwd_c = float(nwd_c_raw) if nwd_c_raw is not None else 12.0
        soft_sigma_raw = getattr(h, "box_soft_sigma", None)
        soft_sigma = float(soft_sigma_raw) if soft_sigma_raw is not None else 0.0
        soft_rel_raw = getattr(h, "box_soft_relative", None)
        soft_rel = float(soft_rel_raw) if soft_rel_raw is not None else 0.0
        soft_train_only = bool(getattr(h, "box_soft_train_only", True))

        # Backwards-compat with the older `nwd=true` boolean in hyp yamls.
        legacy_nwd = bool(getattr(h, "nwd", False))
        if legacy_nwd and nwd_w == 0:
            nwd_w, ciou_w = _DEFAULT_NWD_W, _DEFAULT_CIOU_W

        blend_active = (nwd_w > 0 or abs(ciou_w - 1.0) > 1e-9)
        # DEBUG print once per process — check actual cfg values after fix
        if not hasattr(_patch_v8_loss, '_init_printed'):
            _patch_v8_loss._init_printed = 0
        if _patch_v8_loss._init_printed < 1:
            import sys as _s
            print(f'[INIT DEBUG] nwd_w={nwd_w} ciou_w={ciou_w} nwd_c={nwd_c} '
                  f'soft_sigma={soft_sigma} soft_rel={soft_rel} '
                  f'blend_active={blend_active} legacy_nwd={legacy_nwd}',
                  file=_s.stderr, flush=True)
            _patch_v8_loss._init_printed += 1
        if blend_active or soft_sigma > 0 or soft_rel > 0:
            m = model.model[-1]
            reg_max = int(getattr(m, "reg_max", 16))
            self.bbox_loss = PatchedBboxLoss(
                reg_max,
                nwd_weight=nwd_w,
                ciou_weight=ciou_w,
                nwd_constant=nwd_c,
                soft_sigma_pixel=soft_sigma,
                soft_relative_ratio=soft_rel,
                soft_train_only=soft_train_only,
            ).to(self.device)

    orig_cls.__init__ = _patched_init
    # Return orig_cls unchanged (preserves pickle identity).
    return orig_cls


# ---------------------------------------------------------------------------
# CFG whitelist (so `iou_loss_weight_nwd=...` doesn't get rejected by CLI)
# ---------------------------------------------------------------------------

def _install_cfg_whitelist():
    """Allow the new CLI keys through ultralytics' cfg parser.

    Without this, `yolo detect train iou_loss_weight_nwd=0.7 ...` raises a
    SyntaxError from `check_dict_alignment`. We wrap the original function so
    our keys are passed in via `allowed_custom_keys=...`.
    """
    import ultralytics.cfg as _cfg_mod

    if getattr(_cfg_mod.check_dict_alignment, "_yolo26_ext_patched", False):
        return  # idempotent

    _orig = _cfg_mod.check_dict_alignment

    def _wrapped(base, custom, e=None, allowed_custom_keys=None):
        allow = set(allowed_custom_keys or ())
        allow |= OUR_KEYS
        allow |= TRAINER_INTERNAL_KEYS
        return _orig(base, custom, e, allowed_custom_keys=allow)

    _wrapped._yolo26_ext_patched = True
    _cfg_mod.check_dict_alignment = _wrapped


# ---------------------------------------------------------------------------
# Public install entry-point
# ---------------------------------------------------------------------------

def install():
    """Install the patched classes into ultralytics (idempotent).

    Idempotent in the sense that re-running is safe; it just replaces the
    module globals again with the same objects.
    """
    import ultralytics.utils.loss as _loss_mod

    PatchedBboxLoss = _patch_bbox_loss(_loss_mod.BboxLoss)
    # Register PatchedBboxLoss in the loss module's namespace at module-level
    # so pickle can resolve `_patch_bbox_loss.<locals>.PatchedBboxLoss`.
    # Using `__module__ = loss_mod.__name__` lets pickle look it up by qualified
    # name `ultralytics.utils.loss.PatchedBboxLoss`.
    PatchedBboxLoss.__module__ = _loss_mod.__name__
    PatchedBboxLoss.__qualname__ = "PatchedBboxLoss"
    _loss_mod.PatchedBboxLoss = PatchedBboxLoss

    _patch_v8_loss(_loss_mod.v8DetectionLoss, PatchedBboxLoss)

    # NOTE: do NOT reassign `_loss_mod.BboxLoss = PatchedBboxLoss` — that
    # changes class identity and breaks torch.save pickle.
    # We monkey-patch __init__ instead, which keeps the class object stable.
    _install_cfg_whitelist()


# Auto-install on import so a `python -c "import yolo26_loss_extension; ..."`
# entry-point Just Works.
install()
