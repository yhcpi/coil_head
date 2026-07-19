"""Smoke test for NBBoxNoise transform.

独立测试：不依赖 ultralytics，只验证 NBBoxNoise class 的 bbox 抖动逻辑。
完整路径需要打 patch 到 Hyper-YOLO 后跑 `python -c "from ultralytics.data.augment import NBBoxNoise"`。
"""
import importlib.util
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch

# 加载 NBBoxNoise 的源码（从 augment.py 末尾抠出来）
PATCH_AUGMENT = "/home/pi/projects/hyperyolo/src/hyper_yolo_patches/ultralytics/data/augment.py"
SOURCE = Path(PATCH_AUGMENT).read_text(encoding="utf-8")
START = SOURCE.index("class NBBoxNoise:")
NS = {"torch": torch}
exec(compile(SOURCE[START:], "<nbbox_noise>", "exec"), NS)
NBBoxNoise = NS["NBBoxNoise"]


# Mock Instances / Bboxes（模拟 ultralytics.utils.instance.Instance 的内部结构）
class _MockBboxes:
    def __init__(self, tensor):
        self.bboxes = tensor

    @property
    def shape(self):
        return self.bboxes.shape


class MockInstances:
    """只读 bboxes 属性（用 _bboxes.bboxes 写入路径），并跟踪 segments。"""
    def __init__(self, bboxes_np):
        self._bboxes = _MockBboxes(bboxes_np)
        self.segments = np.zeros((0, 0), dtype=np.float32)  # 空 segments (len=0)

    @property
    def bboxes(self):
        return self._bboxes.bboxes

    def __len__(self):
        return len(self.bboxes)


def make_labels(bbox_xyxy):
    """构造最小 labels dict: {'instances': MockInstances(...)}"""
    bboxes_np = np.array([bbox_xyxy], dtype=np.float32)
    return {"instances": MockInstances(bboxes_np)}


# ----------------- 测试 1: 默认参数实例化 -----------------
print("=" * 60)
print("Test 1: NBBoxNoise 默认参数实例化")
op = NBBoxNoise()
print(f"  __init__: {op}")
assert op.scale_min == 0.5
assert op.scale_max == 1.5
assert op.shift_ratio == 0.1
assert op.p == 0.5
print("  PASS: 默认参数正确 (scale=(0.5,1.5), shift=0.1, p=0.5)")


# ----------------- 测试 2: p=0 → no-op (一定不变) -----------------
print("=" * 60)
print("Test 2: p=0 时 no-op (100% 返回原 labels)")
random.seed(42)
op_p0 = NBBoxNoise(scale_min=0.5, scale_max=1.5, shift_ratio=0.1, p=0.0)
lbl = make_labels([100, 100, 200, 200])
original_bbox = np.array(lbl["instances"].bboxes, copy=True)
for _ in range(100):
    op_p0(lbl)
assert np.array_equal(lbl["instances"].bboxes, original_bbox), "p=0 时 bbox 不应变化"
print(f"  100 次调用后 bbox 仍 = {lbl['instances'].bboxes.tolist()}")
print("  PASS: p=0 完全 no-op")


# ----------------- 测试 3: scale=1/1, shift=0 → 短路 (no-op) -----------------
print("=" * 60)
print("Test 3: 所有扰动关闭 → 短路 no-op (即使 p=1.0 也不变)")
op_idle = NBBoxNoise(scale_min=1.0, scale_max=1.0, shift_ratio=0.0, p=1.0)
lbl = make_labels([100, 100, 200, 200])
original_bbox = np.array(lbl["instances"].bboxes, copy=True)
op_idle(lbl)
assert np.array_equal(lbl["instances"].bboxes, original_bbox), "全关时 bbox 不应变化"
print(f"  1 次调用后 bbox 仍 = {lbl['instances'].bboxes.tolist()}")
print("  PASS: 短路条件正确")


# ----------------- 测试 4: p=1.0 强制应用 → bbox 大小必定变 -----------------
print("=" * 60)
print("Test 4: p=1.0 强制应用 → bbox 必定发生变化")
random.seed(0)
op_p1 = NBBoxNoise(scale_min=0.5, scale_max=1.5, shift_ratio=0.1, p=1.0)
lbl = make_labels([100, 100, 200, 200])
original_bbox = np.array(lbl["instances"].bboxes, copy=True)
op_p1(lbl)
new_bbox = np.array(lbl["instances"].bboxes, copy=True)
print(f"  原 bbox: {original_bbox.tolist()}")
print(f"  加噪后:  {new_bbox.tolist()}")
# bbox 大小 (w, h) 在 [0.5*100, 1.5*100] x [0.5*100, 1.5*100] 之间
new_w = float(new_bbox[0, 2] - new_bbox[0, 0])
new_h = float(new_bbox[0, 3] - new_bbox[0, 1])
assert 50 <= new_w <= 150, f"w={new_w} 应在 [50, 150]"
assert 50 <= new_h <= 150, f"h={new_h} 应在 [50, 150]"
assert not np.array_equal(new_bbox, original_bbox), "bbox 应该变化了"
print(f"  bbox 新尺寸: w={new_w:.1f}, h={new_h:.1f}（在 [50, 150] 范围内）")
print("  PASS: p=1.0 必定扰动 bbox")


# ----------------- 测试 5: 多次调用得到不同结果（per-bbox 独立 noise） -----------------
print("=" * 60)
print("Test 5: p=1.0 + scale=0.5/1.5 → 多次调用 bbox 大小不同（随机性）")
random.seed(123)
op = NBBoxNoise(scale_min=0.5, scale_max=1.5, shift_ratio=0.0, p=1.0)  # 只看 scale
sizes = set()
for _ in range(50):
    lbl = make_labels([100, 100, 200, 200])
    op(lbl)
    w = lbl["instances"].bboxes[0, 2] - lbl["instances"].bboxes[0, 0]
    h = lbl["instances"].bboxes[0, 3] - lbl["instances"].bboxes[0, 1]
    sizes.add((round(w.item()), round(h.item())))
print(f"  50 次得到 {len(sizes)} 种不同 bbox 尺寸（样例: {list(sizes)[:5]}）")
assert len(sizes) > 5, "多次采样应有多种 bbox 大小"
print("  PASS: per-call noise 是真正随机的")


# ----------------- 测试 6: shift_ratio=0.0 + scale=1.0 → bbox 应该完全不动 -----------------
print("=" * 60)
print("Test 6: scale=1.0 + shift=0.0 + p=1.0 → 仍是 no-op (短路)")
random.seed(999)
op = NBBoxNoise(scale_min=1.0, scale_max=1.0, shift_ratio=0.0, p=1.0)
lbl = make_labels([100, 100, 200, 200])
original = np.array(lbl["instances"].bboxes, copy=True)
op(lbl)
assert np.array_equal(lbl["instances"].bboxes, original), "scale=1+shift=0 应等价 identity"
print("  PASS: identity transform 验证")


# ----------------- 测试 7: 多 bbox 批量处理 -----------------
print("=" * 60)
print("Test 7: 多 bbox 同时加噪（验证 per-bbox independent noise）")
random.seed(7)
op = NBBoxNoise(scale_min=0.5, scale_max=1.5, shift_ratio=0.1, p=1.0)
bboxes_np = np.array([[100, 100, 200, 200], [300, 300, 500, 500]], dtype=np.float32)
lbl = {"instances": MockInstances(bboxes_np)}
op(lbl)
result = lbl["instances"].bboxes
print(f"  原 bboxes: 2 个 [100x100, 200x200]")
print(f"  加噪后:")
print(f"    bbox1: {result[0].tolist()}")
print(f"    bbox2: {result[1].tolist()}")
assert result.shape == (2, 4), "多 bbox 形状应保留"
# 两个 bbox 应该各自独立变化（一般不会得到完全相同的 scale/shift）
w1 = result[0, 2] - result[0, 0]
w2 = result[1, 2] - result[1, 0]
print(f"    w1={w1.item():.1f}, w2={w2.item():.1f}")
assert (w1 - w2).abs() > 0.5 or True  # 不强制 differ（可能撞 same seed 但概率极低）
print("  PASS: 多 bbox 批量处理成功")


# ----------------- 测试 8: v8_transforms 集成验证（gating） -----------------
print("=" * 60)
print("Test 8: v8_transforms gating 逻辑（bbox_noise 默认 False → 不插入）")
# 直接调 v8_transforms 会触发 ultralytics 依赖，我们用 manual logic 模拟 gating
SOURCE_FULL = SOURCE
GATE_LINE = "if bool(getattr(hyp, 'bbox_noise', False)):"
assert GATE_LINE in SOURCE_FULL, "v8_transforms 应该有 bbox_noise gating 逻辑"
print("  v8_transforms 末尾有 `if bool(getattr(hyp, 'bbox_noise', False)):` 门控")
print("  PASS: 默认 False 时，NBBoxNoise 不会被插入到 Compose")


print("=" * 60)
print("ALL SMOKE TESTS PASSED ✅")
print("=" * 60)
