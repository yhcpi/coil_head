"""PA-Aug (Physics-Aware Augmentation) 物理感知增强模块

为钢卷头尾小目标检测特化设计的 4 种独立增强组件，每个都是独立可开关的 BaseTransform 子类。
设计上独立于 ultralytics 现有增强管线，通过 cfg 字段 `paaug` 选择启用哪个组件。

4 个组件：
  - motion       ：运动模糊（钢卷在传送带上高速移动时的横向模糊）
  - reflection   ：金属反光（钢卷表面高反光导致的局部过曝/眩光）
  - occlusion    ：轧辊遮挡（机械结构周期性遮挡 tip）
  - noise        ：工业噪声（摄像头电子噪声 + 压缩噪声）

回退机制：
  - paaug=none   ：完全不调用本模块（等价 v4）
  - paaug=motion ：仅启用 motion blur
  - 以此类推

代码影响：augment.py 加 3 行 hook；pa_aug.py 是独立新文件
"""
import math
import random

import cv2
import numpy as np

# 避免循环 import：augment.py 也要 import 本文件
# 直接定义一个最小 BaseTransform stub（如果 augment 已加载则用真的）
try:
    from .augment import BaseTransform
except ImportError:
    class BaseTransform:
        """Standalone BaseTransform stub（pa_aug.py 单测时使用）"""
        def __init__(self):
            pass
        def apply_image(self, labels):
            return labels
        def apply_instances(self, labels):
            return labels
        def apply_semantic(self, labels):
            return labels
        def __call__(self, labels):
            self.apply_image(labels)
            self.apply_instances(labels)
            self.apply_semantic(labels)


# === 组件 1: 运动模糊（钢卷横向移动） ===
class MotionBlur(BaseTransform):
    """模拟钢卷在传送带上高速移动时的横向运动模糊。

    钢卷直径 1-2m，产线速度 1-5 m/s。在 1024 像素画幅中相当于每秒 200-500 像素位移，
    快门时间内可形成 20-50 像素的横向模糊条带。
    """

    def __init__(self, p=0.5, max_kernel=20):
        """Args:
            p: 触发概率（建议 0.5，太高会破坏 tip 形态）
            max_kernel: 模糊核最大长度（像素，1024 坐标系下；默认 20）
        """
        super().__init__()
        self.p = p
        self.max_kernel = max_kernel

    def apply_image(self, labels):
        """对 img 施加随机方向 + 随机长度的运动模糊"""
        img = labels['img']
        if random.random() > self.p:
            return labels
        H, W = img.shape[:2]
        # 钢卷主要横向移动 → 概率 70% 水平、30% 倾斜
        angle = random.choice([0, 0, 0, 15, -15, 30, -30])
        kernel_size = random.randint(5, self.max_kernel)
        kernel = self._motion_kernel(angle, kernel_size)
        img[:] = cv2.filter2D(img, -1, kernel)
        return labels

    @staticmethod
    def _motion_kernel(angle, length):
        """生成指定角度和长度的运动模糊核"""
        kernel = np.zeros((length, length), dtype=np.float32)
        center = length // 2
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        for i in range(length):
            x = int(center + (i - center) * cos_a)
            y = int(center + (i - center) * sin_a)
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1.0
        kernel /= kernel.sum() + 1e-9
        return kernel

    def __call__(self, labels):
        # BaseTransform.__call__ 默认返回 None，会破坏 Compose 流水线；显式返回 labels
        self.apply_image(labels)
        return labels


# === 组件 2: 金属反光（钢卷表面高反光） ===
class MetalReflection(BaseTransform):
    """模拟钢卷表面的局部高反光（过曝/眩光）。

    在工业俯拍场景下，钢卷表面常出现局部过曝的小区域，干扰 tip 检测。
    模拟方法：在图像上随机放置 1-3 个高斯加权的过曝斑块。
    """

    def __init__(self, p=0.3, max_spots=3, max_sigma_ratio=0.08, gain_range=(1.5, 2.5)):
        """Args:
            p: 触发概率
            max_spots: 每张图最多加几个反光斑块
            max_sigma_ratio: 反光斑块最大半径（占图像短边的比例，默认 0.08 ≈ 80 px @ 1024）
            gain_range: 反光斑块中心亮度增益（1.5-2.5 倍）
        """
        super().__init__()
        self.p = p
        self.max_spots = max_spots
        self.max_sigma_ratio = max_sigma_ratio
        self.gain_range = gain_range

    def apply_image(self, labels):
        img = labels['img']
        if random.random() > self.p:
            return labels
        H, W = img.shape[:2]
        n_spots = random.randint(1, self.max_spots)
        for _ in range(n_spots):
            cx = random.randint(0, W)
            cy = random.randint(0, H)
            sigma = random.uniform(self.max_sigma_ratio * 0.3, self.max_sigma_ratio) * min(H, W)
            gain = random.uniform(*self.gain_range)
            # 生成高斯权重图
            x = np.arange(W)
            y = np.arange(H)
            xx, yy = np.meshgrid(x, y)
            gauss = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            # 叠加到图像（加法 + 乘法混合，避免纯白/全黑）
            # 修复：img 是 (H, W, 3)，gauss 是 (H, W) — 加 axis 让广播对齐
            img[:] = np.clip(img.astype(np.float32) * (1.0 + gauss[..., None] * (gain - 1.0)), 0, 255).astype(np.uint8)
        return labels

    def __call__(self, labels):
        # BaseTransform.__call__ 默认返回 None，会破坏 Compose 流水线；显式返回 labels
        self.apply_image(labels)
        return labels


# === 组件 3: 轧辊遮挡（机械结构周期性遮挡） ===
class RollerOcclusion(BaseTransform):
    """模拟钢卷生产线上轧辊/机械结构对视野的周期性遮挡。

    钢卷生产线有多个轧辊/支撑辊，会在视野中形成暗色长条带，遮挡 tip。
    模拟方法：在图像上叠加 1-2 个深色矩形带。
    """

    def __init__(self, p=0.3, max_bars=2, thickness_range=(0.04, 0.10),
                 darken_range=(0.3, 0.6)):
        """Args:
            p: 触发概率
            max_bars: 每张图最多加几个遮挡条
            thickness_range: 遮挡条厚度（占图像短边的比例）
            darken_range: 遮挡条亮度衰减（0.3-0.6 倍原图）
        """
        super().__init__()
        self.p = p
        self.max_bars = max_bars
        self.thickness_range = thickness_range
        self.darken_range = darken_range

    def apply_image(self, labels):
        img = labels['img']
        if random.random() > self.p:
            return labels
        H, W = img.shape[:2]
        n_bars = random.randint(1, self.max_bars)
        for _ in range(n_bars):
            # 70% 横向条带（轧辊上方/下方边缘），30% 倾斜
            horizontal = random.random() < 0.7
            if horizontal:
                thickness = int(random.uniform(*self.thickness_range) * H)
                y0 = random.randint(0, H - thickness)
                darken = random.uniform(*self.darken_range)
                img[y0:y0 + thickness] = (img[y0:y0 + thickness].astype(np.float32) * darken).astype(np.uint8)
            else:
                thickness = int(random.uniform(*self.thickness_range) * W)
                x0 = random.randint(0, W - thickness)
                darken = random.uniform(*self.darken_range)
                img[:, x0:x0 + thickness] = (img[:, x0:x0 + thickness].astype(np.float32) * darken).astype(np.uint8)
        return labels

    def __call__(self, labels):
        # BaseTransform.__call__ 默认返回 None，会破坏 Compose 流水线；显式返回 labels
        self.apply_image(labels)
        return labels


# === 组件 4: 工业噪声（摄像头电子噪声 + 压缩） ===
class IndustrialNoise(BaseTransform):
    """模拟工业摄像头常见的电子噪声 + JPEG 压缩噪声。

    钢卷现场摄像头常因高温/电磁干扰产生椒盐噪声 + 高斯噪声叠加；
    视频压缩/传输又会引入 JPEG block noise。
    """

    def __init__(self, p=0.5, gauss_sigma_range=(3, 10), salt_pepper_p=0.005,
                 jpeg_quality_range=(60, 90)):
        """Args:
            p: 触发概率
            gauss_sigma_range: 高斯噪声 sigma 范围（默认 3-10 / 255）
            salt_pepper_p: 椒盐噪声单像素概率（默认 0.005）
            jpeg_quality_range: JPEG 压缩质量范围（默认 60-90，越低越糊）
        """
        super().__init__()
        self.p = p
        self.gauss_sigma_range = gauss_sigma_range
        self.salt_pepper_p = salt_pepper_p
        self.jpeg_quality_range = jpeg_quality_range

    def apply_image(self, labels):
        img = labels['img']
        if random.random() > self.p:
            return labels

        # 高斯噪声
        sigma = random.uniform(*self.gauss_sigma_range)
        gauss = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img_f = img.astype(np.float32) + gauss
        img[:] = np.clip(img_f, 0, 255).astype(np.uint8)

        # 椒盐噪声
        if self.salt_pepper_p > 0:
            mask = np.random.random(img.shape[:2]) < self.salt_pepper_p
            salt = mask & (np.random.random(img.shape[:2]) < 0.5)
            pepper = mask & ~salt
            img[salt] = 255
            img[pepper] = 0

        # JPEG 压缩（模拟传输损失）
        if self.jpeg_quality_range[0] < 100:
            quality = random.randint(*self.jpeg_quality_range)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buf = cv2.imencode('.jpg', img, encode_param)
            img[:] = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        return labels

    def __call__(self, labels):
        # BaseTransform.__call__ 默认返回 None，会破坏 Compose 流水线；显式返回 labels
        self.apply_image(labels)
        return labels


# === 工厂函数 ===
def make_paaug(component: str):
    """根据 cfg 字段 paaug 字符串构造对应 transform 实例

    Args:
        component: 'none' | 'motion' | 'reflection' | 'occlusion' | 'noise'
    Returns:
        BaseTransform 实例 或 None
    """
    component = (component or 'none').lower()
    if component in ('', 'none', 'false', 'off', '0'):
        return None
    if component == 'motion':
        return MotionBlur(p=0.5, max_kernel=20)
    if component == 'reflection':
        return MetalReflection(p=0.3, max_spots=3, max_sigma_ratio=0.08, gain_range=(1.5, 2.5))
    if component == 'occlusion':
        return RollerOcclusion(p=0.3, max_bars=2)
    if component == 'noise':
        return IndustrialNoise(p=0.5, gauss_sigma_range=(3, 10), salt_pepper_p=0.005)
    raise ValueError(f"Unknown PA-Aug component: {component!r}. "
                     f"Choose from none/motion/reflection/occlusion/noise")
