"""frame_diff_wrapper.py - GUI/批处理用的 FrameDiffProcessor 包装类

封装 detector (FrameDiffCoilDetector) + 截图 (ChangeCapture) 为单帧 API.

设计:
  - detector 与 capture 解耦: detector 算 state/sub_state,
    ChangeCapture 按相对时间点截图, 通过共享 detector 实例通信.
  - 每视频一个子目录: out_dir/{video_name}/.
  - 文件名由 ChangeCapture 内部决定
    (change_XX_RISING_<sec>s.png / change_XX_FALLING_<sec>s.png),
    不可外部覆盖.

依赖 (位于 /home/pi/projects/mm/帧差法/):
  - frame_diff_detector.FrameDiffCoilDetector
  - change_capture.ChangeCapture

典型用法:
    proc = FrameDiffProcessor(out_dir='captures', capture_times_sec=[0.0, 1.0])
    proc.set_video_name('video_001')
    for frame in read_video():
        state, sub, saved = proc.process_frame(frame)
        if saved:
            print(f'  [saved] {saved}')

    # 切换下一段视频
    proc.set_video_name('video_002')
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, '/home/pi/projects/mm/帧差法')

import numpy as np

from frame_diff_detector import FrameDiffCoilDetector
from change_capture import ChangeCapture


class FrameDiffProcessor:
    """钢卷帧差法处理器 = detector + ChangeCapture, 单帧 API.

    单例跨多视频: 用 set_video_name(name) 切换视频 (自动建子目录 + 重建实例).
    同视频内重新初始化用 reset().
    """

    def __init__(
        self,
        out_dir: str,
        capture_times_sec: Optional[List[float]] = None,
    ):
        """
        Args:
            out_dir: 截图输出根目录, 每个视频一个子目录.
            capture_times_sec: CHANGE 段相对偏移时间点列表 (秒).
                默认 [0.0, 1.0] = 切换瞬间 + 持续 1s 各一张.
        """
        self.out_dir = out_dir
        self.capture_times_sec = (
            list(capture_times_sec) if capture_times_sec is not None else [0.0, 1.0]
        )
        os.makedirs(self.out_dir, exist_ok=True)

        self.video_name: Optional[str] = None
        self.capture_dir: Optional[str] = None
        self.det = FrameDiffCoilDetector()
        self.cc: Optional[ChangeCapture] = None

    def set_video_name(self, video_name: str) -> None:
        """切换视频: 建 out_dir/{video_name}/, 重建 detector + ChangeCapture.

        Args:
            video_name: 视频名 (用于子目录名, 不含扩展名).
        """
        self.video_name = video_name
        self.capture_dir = os.path.join(self.out_dir, video_name)
        os.makedirs(self.capture_dir, exist_ok=True)
        self.det = FrameDiffCoilDetector()
        self.cc = ChangeCapture(
            out_dir=self.capture_dir,
            capture_times_sec=self.capture_times_sec,
        )

    def reset(self) -> None:
        """重置 detector + capture 状态机 (保持当前 video_name / 子目录).

        切换视频请用 set_video_name(); 同视频内部状态重置用 reset().
        """
        self.det.reset()
        if self.cc is not None:
            self.cc.reset()

    def process_frame(
        self, frame: np.ndarray, fps: Optional[float] = None
    ) -> Tuple[str, str, Optional[str]]:
        """处理一帧: detector 更新 → ChangeCapture 按需截图.

        Args:
            frame: BGR numpy array, shape (H, W, 3).
            fps: 视频真实帧率. 仅第一次调用时生效 (后续 fps 改变不重置时间窗).
                None = 保持 detector 内部默认值 (60fps, 不推荐用于非 60fps 视频).

        Returns:
            (state, sub_state, saved_path):
              - state: 'STABLE' / 'CHANGE' / 'INIT'
              - sub_state: 'STABLE_NO_COIL' / 'STABLE_COIL' / 'CHANGE_RISING'
                / 'CHANGE_FALLING' / 'INIT'
              - saved_path: 本帧截到的图路径 (None = 本帧无截图);

        Raises:
            RuntimeError: 还未调用 set_video_name() 时.
        """
        if self.cc is None:
            raise RuntimeError(
                "FrameDiffProcessor.process_frame() called before set_video_name(); "
                "call set_video_name(video_name) first."
            )
        # 第一次调用且传了 fps → 覆盖 detector 默认 60fps, 让时间窗归一化正确
        if fps is not None and self.det.frame_idx == 0:
            self.det.fps = float(fps)
        self.det.update(frame)
        saved_path = self.cc.update(self.det, frame)
        return (self.det.state, self.det.sub_state, saved_path)

    def get_current_status(self) -> Dict:
        """当前状态快照: {state, sub_state, frame_idx, fps}.

        Returns:
            dict 包含 detector 实时状态; 调用 process_frame() 后查询才有意义.
        """
        return {
            'state': self.det.state,
            'sub_state': self.det.sub_state,
            'frame_idx': self.det.frame_idx,
            'fps': self.det.fps,
        }
