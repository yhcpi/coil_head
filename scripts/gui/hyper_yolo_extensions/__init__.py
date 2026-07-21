"""Hyper-YOLO 扩展模块 vendored 副本 — 让 v26 best.pt 在官方 ultralytics 8.0.227 下能 load.

问题:
    v26 best.pt 用 Hyper-YOLO 仓库 (repos/Hyper-YOLO) 的 ultralytics 训练,
    pickle GLOBAL 引用 ultralytics.nn.modules.block.MANet. 但:
    - CI runner 上 repos/Hyper-YOLO 不存在 (git 没 track 它, 非 submodule)
    - pip install ultralytics==8.0.227 是纯官方版, 没有 MANet
    → YOLO() 加载报 AttributeError: Can't get attribute 'MANet'

解决:
    把 MANet class (来自 Hyper-YOLO block.py:376) 直接复制到本文件, 然后在
    `import ultralytics` 之后 monkey-patch ultralytics.nn.modules.block.MANet
    = 我们的本地 MANet. pickle 反序列化时能找到类, MANet.__init__ 用的
    Conv/Bottleneck/GroupConv 都已在 ultralytics 标准模块里 (无需 vendored).

Conv/Bottleneck/GroupConv 验证:
    >>> from ultralytics.nn.modules.block import Conv, Bottleneck  # 都存在
    >>> from ultralytics.nn.modules.conv import GroupConv           # 都存在

这个 module 是单文件 vendored, 不依赖任何 git submodule / 外部 repo,
PyInstaller --hidden-import 这个 module 后, .exe bundle 自包含.

源: repos/Hyper-YOLO/ultralytics/nn/modules/block.py:376-397 (commit 时复制)
"""
from __future__ import annotations

import torch
import torch.nn as nn

# 在 import 期间强制 ultralytics 先 load, 这样 Conv/Bottleneck/GroupConv 都在 sys.modules
# 然后我们的 MANet 才能正确继承/使用
import ultralytics.nn.modules.block as _block
import ultralytics.nn.modules.conv as _conv

# 从 ultralytics 标准模块取依赖 (vendored 不重复, 省维护成本)
Conv = _conv.Conv
Bottleneck = _block.Bottleneck
GroupConv = _conv.GroupConv


class MANet(nn.Module):
    """MANet — Hyper-YOLO 自定义模块, vendored 自 block.py:376-397.

    用途: 在 v26 best.pt pickle 反序列化时, 必须能在
    ultralytics.nn.modules.block 命名空间找到 MANet. 我们 monkey-patch
    ultralytics.nn.modules.block.MANet = 这个类, 实现兼容.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, dim_hid, 1, 1),
            GroupConv(dim_hid, dim_hid, kernel_size, 1),
            Conv(dim_hid, self.c, 1, 1),
        )

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = [y0, y1, y2, y3]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv_final(torch.cat(y, 1))


# Monkey-patch 到 ultralytics 命名空间 — 这是关键!
# pickle 反序列化 ultralytics.nn.modules.block.MANet 时会查 sys.modules
_block.MANet = MANet

__all__ = ["MANet", "Conv", "Bottleneck", "GroupConv"]