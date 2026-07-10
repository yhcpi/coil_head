"""Mosaic-Neg: 强制 Mosaic 里至少 1 张是负样本（空 .txt）

继承 Ultralytics Mosaic 类，只重写 get_indexes。

使用方法（在训练前调用，且必须在 import ultralytics.data.dataset 之前）：
    from ultralytics.data import augment
    from ultralytics.data.mosaic_neg import patch_mosaic_neg
    patch_mosaic_neg(neg_p=0.7, neg_min=1, neg_max=2)
"""
import random
from .augment import Mosaic


class MosaicNeg(Mosaic):
    """强制至少 1 张负样本的 Mosaic 变体。"""

    def __init__(self, dataset, imgsz=640, p=1.0, n=4, neg_p=1.0, neg_min=1, neg_max=2):
        super().__init__(dataset=dataset, imgsz=imgsz, p=p, n=n)
        self.neg_p = neg_p
        self.neg_min = neg_min
        self.neg_max = neg_max
        self.neg_indexes = self._collect_neg_indexes(dataset)

    @staticmethod
    def _collect_neg_indexes(dataset):
        """扫描 dataset 找出所有空 .txt 的样本下标。"""
        from pathlib import Path
        neg_idx = []
        n = len(dataset)
        # dataset.label_files 是 list[str]
        label_files = getattr(dataset, 'label_files', None)
        if not label_files:
            return neg_idx
        for i in range(n):
            try:
                lp = Path(label_files[i])
                if not lp.exists() or lp.stat().st_size == 0:
                    neg_idx.append(i)
            except Exception:
                continue
        return neg_idx

    def get_indexes(self, buffer=True):
        """重写：概率 neg_p 下强制至少 1 张负样本。"""
        # 默认行为（保持兼容）。注意：父类在 buffer 空时会崩，我们兜底
        if random.random() > self.neg_p or not self.neg_indexes:
            try:
                return super().get_indexes(buffer=buffer)
            except IndexError:
                # buffer 空，回退到全随机
                return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

        # MosaicNeg 模式
        n_neg = random.randint(
            self.neg_min,
            min(self.neg_max, len(self.neg_indexes), self.n - 1),
        )
        neg_chosen = random.sample(self.neg_indexes, n_neg)

        # 其余位置用随机 / buffer（兜底：buffer 为空就用全范围）
        pool = list(self.dataset.buffer) if (buffer and self.dataset.buffer) else list(range(len(self.dataset)))
        if not pool:
            pool = list(range(len(self.dataset)))
        others = random.choices(pool, k=self.n - 1 - n_neg)

        all_idx = neg_chosen + others
        random.shuffle(all_idx)
        return all_idx


# 默认参数（被 patch 函数读取）
_NEG_P = 0.7
_NEG_MIN = 1
_NEG_MAX = 2


def patch_mosaic_neg(neg_p=0.7, neg_min=1, neg_max=2):
    """monkey-patch 全局 MosaicNeg 参数。"""
    global _NEG_P, _NEG_MIN, _NEG_MAX
    _NEG_P = neg_p
    _NEG_MIN = neg_min
    _NEG_MAX = neg_max


# 默认参数（被 patch 函数读取）
_NEG_P = 0.7
_NEG_MIN = 1
_NEG_MAX = 2


def get_neg_params():
    return _NEG_P, _NEG_MIN, _NEG_MAX