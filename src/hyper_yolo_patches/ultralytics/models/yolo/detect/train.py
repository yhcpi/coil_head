# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import numpy as np
import torch
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils import checks as _checks
_checks.check_amp = lambda *a, **k: False  # 永久跳过 check_amp（避免 bus.jpg 缺失错误）
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, img_path, mode='train', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Construct and return dataloader."""
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # 创新点 v9: 高光抑制模块（默认关闭, hyp 开启时实例化）
        self._install_spec_suppress()
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def _install_spec_suppress(self):
        """Specular highlight suppression 创新点 v9: 把 SpecSuppress 插入 backbone 出口.

        - 默认关闭 (self.args.spec_suppress=False)，hyp 开启时实例化
        - 注入位置: backbone 末层（SPPF, index=9 for yolov8-p2）后; neck 入口前
        - 用 register_forward_pre_hook 在 Sequential 入口捕获 3 通道原图, 注入 spec
        - v8DetectionLoss 通过 _parent_model_ref 找到模块并叠加 recon loss
        """
        from ultralytics.nn.modules import SpecSuppress
        import torch.nn as nn

        if not getattr(self.args, 'spec_suppress', False):
            return  # 默认关闭，对 best.pt 推理零影响

        # backbone 出口层: yolov8-p2 中 SPPF 在 model.model[9]
        backbone_exit_idx = 9
        inner = de_parallel(self.model)
        if not hasattr(inner, 'model') or len(inner.model) <= backbone_exit_idx:
            LOGGER.warning(f"[spec_suppress] model has no layer {backbone_exit_idx}, skip")
            return

        # 通道数: SPPF 输出的 c2 (n-scale: 1024*0.25=256)
        sppf = inner.model[backbone_exit_idx]
        c_out = None
        cv2 = getattr(sppf, 'cv2', None)
        if cv2 is not None and hasattr(cv2, 'conv'):
            c_out = cv2.conv.out_channels
        if not isinstance(c_out, int) or c_out <= 0:
            c_out = 256  # n-scale P5 fallback

        # 重建 Sequential, 把 SpecSuppress 插入到 backbone_exit_idx+1 位置
        # 关键: 同时偏移后续 head 层的 i / f / save 引用, 否则 Concat 索引错位
        spec = SpecSuppress(c=c_out, use_recon=True).to(self.device)
        modules = list(inner.model.children())
        insert_idx = backbone_exit_idx + 1
        modules.insert(insert_idx, spec)
        inner.model = nn.Sequential(*modules)
        spec.i = insert_idx
        spec.f = -1
        spec.type = 'SpecSuppress'

        for i in range(insert_idx + 1, len(modules)):
            m = modules[i]
            if hasattr(m, 'i'):
                m.i = i
            if hasattr(m, 'f') and m.f != -1:
                if isinstance(m.f, list):
                    m.f = [j + 1 if j >= insert_idx else j for j in m.f]
                elif isinstance(m.f, int) and m.f >= insert_idx:
                    m.f = m.f + 1
        # save 列表同步偏移
        if hasattr(inner, 'save'):
            inner.save = sorted({s + 1 if s >= insert_idx else s for s in inner.save})
        LOGGER.info(
            f"[spec_suppress] Installed SpecSuppress(c={c_out}) at idx {insert_idx}, "
            f"new model len={len(inner.model)}, save={inner.save[:5]}..."
        )

        # forward_pre_hook on DetectionModel: 捕获 3 通道原图, 注入 spec
        # 注意 1: 不能 hook SpecSuppress 自身, 那样拿到的是 256 通道 SPPF 输出
        # 注意 2: 也不能 hook Sequential (inner.model), 因为 DetectionModel._predict_once
        #         直接遍历 Sequential 不调 self.model(x), Sequential pre_hook 不触发
        def _inject_img_ctx(_model, inp):
            x = inp[0] if isinstance(inp, tuple) else inp
            if isinstance(x, torch.Tensor) and x.dim() == 4 and x.shape[1] == 3:
                spec.set_image_ctx(x)
            # 其他情况 (4 通道 mosaic+edge) 跳过, recon loss 不参与

        inner.register_forward_pre_hook(_inject_img_ctx)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return yolo.detect.DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(images=batch['img'],
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir, on_plot=self.on_plot)


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize YOLO model given training data and device."""
    model = cfg.model
    data = cfg.data  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    # Fix: pass the full cfg (not just 3 keys) so epochs/batch/name/imgsz are preserved.
    overrides = {k: v for k, v in vars(cfg).items() if v is not None}
    overrides['model'] = model
    overrides['data'] = data
    overrides['device'] = device
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**overrides)
    else:
        trainer = DetectionTrainer(overrides=overrides)
        trainer.train()


if __name__ == '__main__':
    from ultralytics.cfg import entrypoint
    entrypoint()  # 读 sys.argv，不再 hardcode debug（保留 model/data/epochs 等 args）
