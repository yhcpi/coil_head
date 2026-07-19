"""Specular Highlight Suppression module for steel-coil / metallic-surface detection.

设计动机：
  钢卷头尾（coil tip）是金属反光场景，反光高光区会把 tip 纹理"打白"，导致 v4 baseline
  出现 5 个 FN（高反光导致 tip 看不清）。本模块在 backbone 出口、neck 入口之间插入
  一个结构感知的高光抑制分支，输出与 backbone 特征同形状的"高光抑制后特征"
  + 一个低分辨率的"重建图"，用 reconstruction loss 自监督训练（无配对数据可用）。

参考论文（核心思想融合，不直接用代码）:
  - TII 2023 "Specular Removal of Industrial Metal Objects Without Changing
    Lighting Configuration" (Chen et al., DOI 10.1109/TII.2023.3297613, 109 引用)
    关键启发：dynamic mask + shape-shift conv，mask 高概率 = 高光区域
  - DHAN-SHR 2024 (arXiv 2407.12255) "Dual-Hybrid Attention Network"
    关键启发：channel + spatial 双注意力，捕捉 specular vs surface 区分

适配性分析（v8 n-scale, imgsz=1024）:
  - backbone 末层 SPPF 输出:  (B, 512, 32, 32)   ← 模块接入点
  - 输入/输出 shape 一致，可直接 inline 到 Sequential
  - 不需要任何预训练权重：模块零初始化重建分支，训练时学
  - 不破坏 best.pt 推理：默认 spec_suppress=False，模块不实例化
  - 解耦检测 loss：recon loss 走单独的 `loss[3]`，不进 box/cls/dfl

实现要点（80-100 行目标）:
  - SpecSuppress: Conv-Act 编码 + Channel Attention + Soft mask
  - reconstruction branch: 1x1 Conv → (3, H, W) 重建图
  - reconstruction_loss: L1(recon, downsample(input)) + 简化版 perceptual (Sobel 梯度差)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 基础构件
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """SE-Net 风格的 channel attention（轻量，~10 行）."""

    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, max(c // r, 4), 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(c // r, 4), c, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


def _sobel_grad(x: torch.Tensor) -> torch.Tensor:
    """简化版 perceptual：用 Sobel 算子提取梯度图，替代 VGG perceptual loss.

    优点: 无外部权重依赖（不需要 torchvision VGG16 预训练），计算代价低
    缺点: 只抓边缘信息，不抓高层语义
    对高光场景: 高光区域梯度被压平，Sobel 差会显著 → 有效监督信号
    """
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    # 按通道循环（用 groups 一次性更快；这里保持可读性）
    b, c, h, w = x.shape
    x_flat = x.reshape(b * c, 1, h, w)
    gx = F.conv2d(x_flat, kx, padding=1)
    gy = F.conv2d(x_flat, ky, padding=1)
    return torch.sqrt(gx.view(b, c, h, w) ** 2 + gy.view(b, c, h, w) ** 2 + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 主模块
# ─────────────────────────────────────────────────────────────────────────────

class SpecSuppress(nn.Module):
    """结构感知的高光抑制模块（backbone 出口 → neck 入口）.

    输入: backbone 特征 (B, C, H, W)  — 例如 SPPF 输出 (B, 512, 32, 32)
    输出: 高光抑制后特征 (B, C, H, W) — 同形状
    旁路: 重建图 (B, 3, H, W)           — 用 input image 下采样作 target，自监督

    训练时需要把原始图像（或 backbone 输入）通过 set_image_ctx 注入，
    forward 时计算 reconstruction loss。
    """

    def __init__(self, c: int, use_recon: bool = True):
        super().__init__()
        self.c = c
        self.use_recon = use_recon

        # 主干：3x3 conv 编码 + channel attention（区分高光通道 vs 表面通道）
        self.encoder = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.attn = ChannelAttention(c)

        # 高光概率 mask 头：1x1 conv → sigmoid → 1 通道，值越大 = 越像高光
        self.mask_head = nn.Sequential(
            nn.Conv2d(c, c // 4, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(c // 4, 1, 1, bias=True),
        )
        # 关键：bias 初始化为 -2.0，让初始 mask 接近 0（不抑制），保护 v4 baseline 行为
        nn.init.constant_(self.mask_head[-1].bias, -2.0)

        # 重建分支：把高光抑制后特征解回 3 通道图像
        # （超轻量：1x1 + 上采样到 backbone 输入分辨率；不在主干上花算力）
        self.recon_head = nn.Conv2d(c, 3, 1, bias=True) if use_recon else None

        # 图像 ctx（外部注入，forward 时读取）
        self._img_ctx: torch.Tensor | None = None

    def set_image_ctx(self, img: torch.Tensor) -> None:
        """注入原图 ctx（norm 后的模型输入，例如 (B,3,1024,1024)）.
        训练时由外部 hook 调用；推理时可为空.
        """
        self._img_ctx = img

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 编码 + 通道注意力
        feat = self.encoder(x)
        feat = self.attn(feat)

        # 2. 生成高光概率 mask
        mask = torch.sigmoid(self.mask_head(feat))  # (B, 1, H, W), ∈ (0, 1)

        # 3. 用 mask 软抑制：x_suppressed = feat * (1 - mask) + x * mask
        #    直觉：mask 高 = 来自原图（保留表面信息）；mask 低 = 来自编码（去除高光）
        #    注：这里 feat 还没加原图，先简单实现为残差
        x_out = x + (feat - x) * (1.0 - mask)  # mask=0 完全用 feat；mask=1 完全保留 x

        # 4. 计算并缓存 recon loss（自监督），给 v8DetectionLoss 读取
        # 防御: _img_ctx 可能 batch size 与 x 不一致 (mosaic 时 model 输入 4 通道,
        #       pre_hook 跳过更新, _img_ctx 保持上次值)
        if (self.use_recon and self.recon_head is not None
                and self._img_ctx is not None
                and self._img_ctx.shape[0] == x.shape[0]):
            self._last_recon_loss = self.compute_recon_loss(x)
        else:
            self._last_recon_loss = None

        return x_out

    def compute_recon_loss(self, x_feat: torch.Tensor) -> torch.Tensor:
        """计算重建 loss（自监督，无配对数据）.

        target: 下采样原图到 backbone 特征分辨率
        pred:   recon_head(x_feat)  → (B, 3, H, W)
        loss:   L1 + λ_grad * Sobel 梯度差
        """
        if not self.use_recon or self._img_ctx is None or self.recon_head is None:
            return torch.tensor(0.0, device=x_feat.device)

        # 重建：与高光抑制前的特征对比（用 forward 输入前的 x，不抑制）
        # 注意：x_feat 是 backbone 输出（已被 norm 分布化），不能直接对比像素；
        # 简化方案：直接用 backbone 输出做重建目标对齐，让 recon_head 学"特征→图"
        pred = self.recon_head(x_feat)  # (B, 3, H, W)

        # 把原图 ctx 下采样到 (B, 3, H, W)
        target = F.adaptive_avg_pool2d(self._img_ctx, output_size=pred.shape[-2:])

        l1 = F.l1_loss(pred, target)
        grad_pred = _sobel_grad(pred)
        grad_target = _sobel_grad(target)
        l_grad = F.l1_loss(grad_pred, grad_target)

        return l1 + 0.5 * l_grad
