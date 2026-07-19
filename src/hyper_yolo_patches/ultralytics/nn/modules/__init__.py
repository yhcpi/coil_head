# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
这个文件是做什么的？
__init__.py 文件有两个核心作用：

作用	解释
① 标记文件夹为 Python 包	告诉 Python："这个文件夹是一个模块包，可以被 import"
② 导出可供外部使用的模块	相当于一个"目录"，列出这个包对外提供哪些功能
当你写 from ultralytics.nn.modules import Conv, C2f 时，Python 就是通过 __init__.py 知道 Conv 和 C2f 在哪里可以找到。
"""

from .block import (C1, C2, C3, MANet, HyperComputeModule, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f,
                    C3Ghost, C3x, GhostBottleneck, HGBlock, HGStem, Proto, RepC3, ResNetLayer)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, GroupConv, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)
# ── 本项目创新点：钢卷反光抑制模块 ──
# 默认不在 yaml 中实例化；通过 hyp_v9_spec_suppress.yaml: spec_suppress=True 启用
from .spec_suppress import SpecSuppress

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 'ResNetLayer',
           'MANet', 'HyperComputeModule', 'GroupConv', 'SpecSuppress')
