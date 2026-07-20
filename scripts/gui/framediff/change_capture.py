"""change_capture.py - CHANGE 状态截图工具 (与 detector 解耦)

功能:
  detector 进入 CHANGE 状态后, 在指定的相对时间点截取 (默认 0s 和 1s 各 1 张).
  - 进入 CHANGE 瞬间: 立即截第 1 张 (0s)
  - 持续 1s: 截第 2 张 (1s)
  - 离开 CHANGE 时: 处理已有截图

  RISING 段 (STABLE_NO_COIL → STABLE_COIL): 全部保留 (这一定是真 coil 来)
  FALLING 段 (STABLE_COIL → ?):
    - 离开到 STABLE_NO_COIL → 真 LEAVE, 全部保留
    - 离开到 STABLE_COIL   → 假 LEAVE 被 False LEAVE Recovery 撤销, 全部丢弃

  文件命名按子状态分段:
    change_<idx>_RISING_<sec>s.png   ← RISING 段按相对偏移命名
    change_<idx>_FALLING_<sec>s.png  ← FALLING 段按相对偏移命名

与 frame_diff_detector.py 完全解耦:
  - 只读 detector.state / sub_state / frame_idx / fps, 不修改 detector
  - 接受任意 "frame 图像" (numpy array), 内部用 cv2.imwrite 保存

典型用法:
    cap = ChangeCapture(out_dir='captures')  # 默认 [0, 1]
    while True:
        ret, frame = cap.read()
        det.update(frame, fps=fps)
        saved = cap.update(det, frame)
        if saved:
            print(f'  [capture] {saved}')
"""
import os
import cv2
from typing import Optional, List


class ChangeCapture:
    """CHANGE 状态截图工具 (按相对时间点截取)"""

    def __init__(self, out_dir: str, capture_times_sec: Optional[List[float]] = None):
        """
        Args:
            out_dir: 输出文件夹
            capture_times_sec: 截取时间点列表 (相对 CHANGE 段开始, 秒)
                默认 [0.0, 1.0] = 切换瞬间 + 持续 1s 各截 1 张, 共 2 张
        """
        self.out_dir = out_dir
        self.capture_times_sec = sorted(capture_times_sec) if capture_times_sec else [0.0, 1.0]
        os.makedirs(out_dir, exist_ok=True)

        # 状态机
        self._phase = 'NONE'  # NONE / IN_RISING / IN_FALLING
        self._segment_start_frame = 0
        self._fps = 30.0
        self._pending: List[tuple] = []  # [(frame_idx, frame_copy, offset_sec, phase)]
        self._saved_count = 0
        self.saved_paths: List[str] = []

    def update(self, det, frame) -> Optional[str]:
        """每帧调用一次. 返回保存的路径, 或 None."""
        state = det.state
        sub = det.sub_state
        cur_frame = det.frame_idx
        fps = getattr(det, 'fps', 30.0)
        self._fps = fps

        # === STABLE_NO_COIL → CHANGE_RISING: 启动 RISING 段 ===
        if self._phase == 'NONE' and sub == 'CHANGE_RISING':
            self._phase = 'IN_RISING'
            self._segment_start_frame = cur_frame
            self._pending = []
            return None

        # === STABLE_COIL → CHANGE_FALLING: 启动 FALLING 段 ===
        if self._phase == 'NONE' and sub == 'CHANGE_FALLING':
            self._phase = 'IN_FALLING'
            self._segment_start_frame = cur_frame
            self._pending = []
            return None

        # === IN_RISING / IN_FALLING: 按 capture_times_sec 截取 ===
        if self._phase in ('IN_RISING', 'IN_FALLING'):
            if sub.startswith('CHANGE'):
                # pending 元素: (capture_idx, frame_copy, offset_sec, phase)
                captured_idx_set = {p[0] for p in self._pending}
                for idx, t_sec in enumerate(self.capture_times_sec):
                    if idx in captured_idx_set:
                        continue  # 该时间点已截过
                    target_frame = self._segment_start_frame + int(t_sec * fps)
                    if target_frame <= cur_frame:
                        offset_sec = (cur_frame - self._segment_start_frame) / fps
                        self._pending.append((idx, frame.copy(), offset_sec, self._phase))
                        break  # 每帧最多截一张
                return None
            else:
                # 离开 CHANGE → 处理 pending
                return self._flush_pending(sub)

        return None

    def _flush_pending(self, exit_sub: str) -> Optional[str]:
        """CHANGE 段结束, 决定写盘还是丢弃."""
        if not self._pending:
            self._phase = 'NONE'
            return None

        # RISING 段 → 离开到 STABLE_COIL 是正常完成
        # FALLING 段 → 离开到 STABLE_NO_COIL 才是真 LEAVE
        if self._phase == 'IN_RISING':
            keep = (exit_sub == 'STABLE_COIL')
        else:  # IN_FALLING
            keep = (exit_sub == 'STABLE_NO_COIL')

        last_path = None
        if keep:
            for cur_frame, f, offset_sec, phase in self._pending:
                sub_short = 'RISING' if phase == 'IN_RISING' else 'FALLING'
                fname = f'change_{self._saved_count:02d}_{sub_short}_{offset_sec:05.2f}s.png'
                path = os.path.join(self.out_dir, fname)
                cv2.imwrite(path, f)
                self.saved_paths.append(path)
                last_path = path
                self._saved_count += 1

        self._pending = []
        self._phase = 'NONE'
        return last_path

    def reset(self):
        """重置状态机"""
        self._phase = 'NONE'
        self._segment_start_frame = 0
        self._pending = []
        self._saved_count = 0
        self.saved_paths = []


def collect_videos(root_dir: str, exts=('.mp4', '.MP4', '.avi', '.mov', '.mkv')) -> List[str]:
    """递归收集文件夹下所有视频, 返回排序后的绝对路径列表"""
    root_dir = os.path.abspath(root_dir)
    result = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(exts):
                result.append(os.path.join(dirpath, fn))
    result.sort()
    return result