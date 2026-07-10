# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from .metrics import bbox_iou
from .tal import bbox2dist


def wasserstein_loss(pred_bboxes, target_bboxes, eps=1e-7, constant=12.0):
    """Normalized Wasserstein Distance (NWD) between predicted and target bboxes.

    Models each bbox as a 2D Gaussian N(mu, Sigma) where Sigma = diag((w/2)^2, (h/2)^2),
    then uses the 2nd-order Wasserstein distance and exponentiates it for normalization.
    Robust to position deviation and box-shape ambiguity — ideal for tiny or
    loosely-annotated objects (steel coil ends, etc.).

    Args:
        pred_bboxes: (N, 4) in xyxy format (predicted decoded boxes).
        target_bboxes: (N, 4) in xyxy format.
        constant: dataset-dependent scaling factor. Paper uses 12.0 for AI-TOD
            (Wang et al. 2021). Tune if your avg box size differs significantly.

    Returns:
        nwd (Tensor): similarity in [0, 1], 1 = identical, ~0 = far apart.
            Loss is computed as `1 - nwd` by the caller.
    """
    b1_x1, b1_y1 = pred_bboxes[:, 0], pred_bboxes[:, 1]
    b1_x2, b1_y2 = pred_bboxes[:, 2], pred_bboxes[:, 3]
    b2_x1, b2_y1 = target_bboxes[:, 0], target_bboxes[:, 1]
    b2_x2, b2_y2 = target_bboxes[:, 2], target_bboxes[:, 3]

    center1_x = (b1_x1 + b1_x2) / 2
    center1_y = (b1_y1 + b1_y2) / 2
    center2_x = (b2_x1 + b2_x2) / 2
    center2_y = (b2_y1 + b2_y2) / 2

    center_distance = (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2 + eps

    w1 = b1_x2 - b1_x1 + eps
    h1 = b1_y2 - b1_y1 + eps
    w2 = b2_x2 - b2_x1 + eps
    h2 = b2_y2 - b2_y1 + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
    wasserstein = torch.sqrt(center_distance + wh_distance)
    return torch.exp(-wasserstein / constant)


def gwd_loss(pred_bboxes, target_bboxes, eps=1e-7, tau=1.0):
    """Gaussian Wasserstein Distance (GWD) between predicted and target bboxes.

    Variant of NWD that uses the raw Wasserstein distance (no exp normalization).
    Loss is `1 - exp(-gwd / tau)` per the GWD paper (Yang et al. 2022, "GWD: A
    Novel Method for Small Object Detection"). The tau controls smoothness of
    the gradient w.r.t. distance. Use this for tiny / loosely-annotated bboxes
    when NWD's exp normalization saturates (loss near 0 even for wrong boxes).

    Args:
        pred_bboxes: (N, 4) xyxy.
        target_bboxes: (N, 4) xyxy.
        tau: smoothness factor (paper uses 1.0; smaller = sharper gradient near 0).

    Returns:
        gwd (Tensor): distance value per pair, ≥ 0. Loss = 1 - exp(-gwd/tau).
    """
    b1_x1, b1_y1 = pred_bboxes[:, 0], pred_bboxes[:, 1]
    b1_x2, b1_y2 = pred_bboxes[:, 2], pred_bboxes[:, 3]
    b2_x1, b2_y1 = target_bboxes[:, 0], target_bboxes[:, 1]
    b2_x2, b2_y2 = target_bboxes[:, 2], target_bboxes[:, 3]

    center1_x = (b1_x1 + b1_x2) / 2
    center1_y = (b1_y1 + b1_y2) / 2
    center2_x = (b2_x1 + b2_x2) / 2
    center2_y = (b2_y1 + b2_y2) / 2

    center_distance = (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2 + eps

    w1 = b1_x2 - b1_x1 + eps
    h1 = b1_y2 - b1_y1 + eps
    w2 = b2_x2 - b2_x1 + eps
    h2 = b2_y2 - b2_y1 + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
    wasserstein = torch.sqrt(center_distance + wh_distance)
    return wasserstein


def coverage_loss(pred_bboxes, target_bboxes, sigma=20.0, eps=1e-6):
    """Coverage Loss: -log(P(GT_center ∈ predicted_bbox))。

    物理含义：
      - 预测 bbox 包住 GT 中心 → loss ≈ 0（达成"覆盖"目标）
      - 预测 bbox 在 GT 中心外 → loss 大（没覆盖）
    与 IoU/NWD 互补：
      - IoU/NWD 要求 bbox 大小接近 → 宽容 GT 下 loss 失真
      - Coverage 只看"中心是否落入" → 匹配"宽容标注包住目标"的语义

    数学：
      pred_x1, pred_y1, pred_x2, pred_y2 是预测 bbox 的 4 边
      gt_cx, gt_cy 是 GT 中心
      in_x = sigmoid((gt_cx - pred_x1) / sigma) - sigmoid((gt_cx - pred_x2) / sigma)
      ↑ GT 中心在 pred_x1 右侧且 pred_x2 左侧的概率
      类似 in_y
      coverage = in_x * in_y  ← 联合落入概率
      loss = -log(coverage)

    sigma 是软化参数：
      - sigma=10: 严苛，5px 边界外就强烈衰减
      - sigma=20: 中等（推荐起点，匹配 ~30-50px 的小目标）
      - sigma=50: 极松，几乎只看中心点是否在 bbox 内

    Returns:
        loss (Tensor): 平均 -log(coverage)，单位 nats
    """
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_bboxes.unbind(-1)
    gt_cx = (target_bboxes[..., 0] + target_bboxes[..., 2]) / 2
    gt_cy = (target_bboxes[..., 1] + target_bboxes[..., 3]) / 2

    in_x = torch.sigmoid((gt_cx - pred_x1) / sigma) - \
           torch.sigmoid((gt_cx - pred_x2) / sigma)
    in_y = torch.sigmoid((gt_cy - pred_y1) / sigma) - \
           torch.sigmoid((gt_cy - pred_y2) / sigma)
    coverage = in_x * in_y
    # 2026-07-09 Fix v2: in_x / in_y ∈ (-1, 1)，当 pred bbox 完全不覆盖 GT 时
    # (sigmoid(a)-sigmoid(b)) 会变成 (-1, 0)，coverage = 负数，log(负) = NaN。
    # 取绝对值 + clamp 保证 coverage ∈ [eps, 1]（log 安全）。
    coverage = coverage.abs().clamp(min=eps)
    return -torch.log(coverage)


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).mean(1).sum()
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, ):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False, nwd=False, nwd_constant=12.0,
                 gwd=False, gwd_tau=1.0,
                 coverage=False, coverage_weight=0.5, coverage_sigma=20.0,
                 looseness_alpha=0.0, looseness_target_area=400.0, stride=32.0):
        """Initialize the BboxLoss module with regularization maximum and DFL settings.

        Args:
            coverage: 若 True，叠加 Coverage Loss 作为 bbox loss 的附加项
            coverage_weight: Coverage Loss 的权重（默认 0.5，与 IoU 等量级）
            coverage_sigma: Coverage Loss 的 sigmoid 软化参数（像素，imgsz 坐标系下）
            looseness_alpha: 贝叶斯先验 / 宽松度自适应 bbox loss 权重。
                looseness = max(1.0, bbox_area / looseness_target_area)。
                weight *= 1 / looseness^looseness_alpha。
                =0.0 → 完全关闭（v4 行为）；>0 → 宽松 GT 的 bbox loss 权重衰减
            looseness_target_area: 估计真实目标面积（像素²），默认 400 = 20×20 px tip
            stride: 模型 stride（默认 32，v8DetectionLoss 调用时覆盖）。用于把
                特征空间 bbox 面积乘 stride² 还原到 imgsz 像素空间，跟
                looseness_target_area 单位一致——否则坐标空间错配会导致整个
                looseness 块成为 silent no-op（clamp(min=1.0) 永远生效）。
        """
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.nwd = nwd
        self.nwd_constant = nwd_constant
        self.gwd = gwd
        self.gwd_tau = gwd_tau
        self.coverage = coverage
        self.coverage_weight = coverage_weight
        self.coverage_sigma = coverage_sigma
        self.looseness_alpha = looseness_alpha
        self.looseness_target_area = looseness_target_area
        self.stride = float(stride)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss or NWD loss (Wang et al. 2021, https://arxiv.org/abs/2110.13389).

        When self.nwd=True, replaces IoU-based regression with Normalized Wasserstein
        Distance — robust to position deviation and box-shape ambiguity, which is critical
        for tiny / loosely-annotated objects (e.g. steel coil ends).

        When self.coverage=True, additionally adds Coverage Loss (pred bbox 包住 GT 中心)
        as a soft auxiliary signal — does not replace IoU/NWD, just adds to it.

        When self.looseness_alpha>0, applies Bayesian prior on bbox loss weight:
        宽松 GT（bbox 面积远大于真实目标）权重衰减，模型少学错位 bbox，
        把学习压力放到 cls / conf 通道。
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        # 贝叶斯先验 / 宽松度自适应 bbox loss 权重
        # looseness = bbox_area / target_area (clamp >= 1)
        # weight *= looseness^(-alpha)，looseness 越大权重越小
        if self.looseness_alpha > 0 and fg_mask.any():
            tb = target_bboxes[fg_mask]
            tb_w = (tb[..., 2] - tb[..., 0]).clamp(min=1.0)
            tb_h = (tb[..., 3] - tb[..., 1]).clamp(min=1.0)
            tb_area = tb_w * tb_h
            # Fix: target_bboxes 已被 /stride_tensor 进入特征空间；乘 stride² 还原到 imgsz 像素²，
            # 跟 looseness_target_area 单位一致——否则坐标空间错配 → silent no-op
            tb_area_pixel = tb_area * (self.stride ** 2)
            looseness = (tb_area_pixel / max(self.looseness_target_area, 1.0)).clamp(min=1.0)
            weight = weight * (looseness.unsqueeze(-1) ** (-self.looseness_alpha))
        if self.gwd:
            # Gaussian Wasserstein Distance (raw, no exp) — loss = 1 - exp(-gwd/tau)
            gwd_val = gwd_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], tau=self.gwd_tau)
            gwd_sim = torch.exp(-gwd_val / max(self.gwd_tau, 1e-9))
            loss_iou = ((1.0 - gwd_sim) * weight).sum() / target_scores_sum
        elif self.nwd:
            nwd_val = wasserstein_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask],
                                       constant=self.nwd_constant)
            loss_iou = ((1.0 - nwd_val) * weight).sum() / target_scores_sum
        else:
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # Coverage Loss 叠加项（默认关闭；通过 hyp_aug.yaml 的 coverage: true 启用）
        if self.coverage and fg_mask.any():
            # 2026-07-09 Fix: 跟 IoU 一样的加权归一化方式，避免单实例数据集爆炸
            # cov_per_anchor 形状 [N]，weight 形状 [N, 1]，squeeze(-1) 对齐
            cov_per_anchor = coverage_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask],
                                            sigma=self.coverage_sigma)
            cov_val = (cov_per_anchor * weight.squeeze(-1)).sum() / target_scores_sum
            loss_iou = loss_iou + self.coverage_weight * cov_val

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        # Pass through NWD switch from hyperparameters (set via `model.args.nwd`)
        nwd_on = bool(getattr(h, 'nwd', False))
        nwd_c = float(getattr(h, 'nwd_constant', 12.0))
        # GWD switch (set via `model.args.gwd`); 与 nwd 互斥，gwd 优先
        gwd_on = bool(getattr(h, 'gwd', False))
        gwd_t = float(getattr(h, 'gwd_tau', 1.0))
        # Coverage Loss switch (set via `model.args.coverage`); 默认关
        coverage_on = bool(getattr(h, 'coverage', False))
        coverage_w = float(getattr(h, 'coverage_weight', 0.5))
        coverage_s = float(getattr(h, 'coverage_sigma', 20.0))
        # 贝叶斯先验 / 宽松度自适应 bbox loss 权重（默认关 = v4 行为）
        looseness_alpha = float(getattr(h, 'looseness_alpha', 0.0))
        looseness_target_area = float(getattr(h, 'looseness_target_area', 400.0))
        # Fix: 把模型 stride 传给 BboxLoss，用于把特征空间 bbox 面积还原到像素空间，
        # 否则 looseness_target_area 单位错配 → silent no-op
        bbox_loss_stride = float(m.stride.mean()) if hasattr(m, 'stride') else 32.0
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl,
                                  nwd=nwd_on, nwd_constant=nwd_c,
                                  gwd=gwd_on, gwd_tau=gwd_t,
                                  coverage=coverage_on,
                                  coverage_weight=coverage_w,
                                  coverage_sigma=coverage_s,
                                  looseness_alpha=looseness_alpha,
                                  looseness_target_area=looseness_target_area,
                                  stride=bbox_loss_stride).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)
            # Masks loss
            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            loss[1] = self.calculate_segmentation_loss(fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto,
                                                       pred_masks, imgsz, self.overlap)

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor,
                         area: torch.Tensor) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i],
                                              marea_i[fg_mask_i])

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(fg_mask, target_gt_idx, keypoints, batch_idx,
                                                             stride_tensor, target_bboxes, pred_kpts)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes,
                                 pred_kpts):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros((batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]),
                                        device=keypoints.device)

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, :keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2]))

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='mean')
        loss_items = loss.detach()
        return loss, loss_items
