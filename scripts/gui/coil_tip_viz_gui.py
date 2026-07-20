"""coil_tip_viz_gui.py - 钢卷头尾检测可视化主 GUI 程序

依赖模块 (已实现, 直接 import):
    frame_diff_wrapper.FrameDiffProcessor  (同目录)
    hyper_inference.HyperYoloDetector       (同目录)
    pyav_reader.open_video                  (/home/pi/projects/mm/帧差法/)

UI 布局 (tkinter, 1280x800):
    顶部工具栏 (height=50):  [选择文件夹] [开始/暂停] | 当前视频名 + 进度
    左侧视频列表 (width=30):  Listbox 列出扫描到的视频文件
    中央视频显示:             Label 显示当前帧 (含 state/bbox 叠加)
    右侧状态面板 (width=300): state/sub_state/FPS + 检测结果 + 截图列表
    底部控制 (height=30):     [上一段] [下一段] | 段计数

线程模型:
    - 主线程: tkinter event loop (root.mainloop())
    - 后台 1 个 worker 线程: 读帧 → FrameDiffProcessor.process_frame() → HyperYoloDetector.detect()
                              通过 queue.Queue 推 (frame_bgr, status) 给主线程
    - 主线程每 100ms 用 root.after(100, self.poll_queue) 消费队列

启动:
    python /home/pi/projects/hyperyolo/scripts/gui/coil_tip_viz_gui.py
"""
import os
import sys
import time
import queue
import threading
from typing import Optional, List, Dict, Any

import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

# 同目录模块
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
from frame_diff_wrapper import FrameDiffProcessor
from hyper_inference import HyperYoloDetector

# pyav_reader (海康 IMKH 等 cv2 不识别的格式可回退)
sys.path.insert(0, '/home/pi/projects/mm/帧差法')
from pyav_reader import open_video


# ----------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------
CAPTURE_ROOT = '/home/pi/projects/hyperyolo/runs/captures_gui'
CONF_HI = 0.5  # bbox 颜色分界: > CONF_HI 绿色, 否则黄色
POLL_MS = 100  # 主线程 poll queue 间隔
IMG_MAX_W, IMG_MAX_H = 880, 680  # 中央 Label 显示上限 (留出 toolbar/status 余量)
VIDEO_EXTS = ('.mp4', '.MP4', '.avi', '.AVI', '.mkv', '.MOV',
              '.mov', '.flv', '.ts', '.m4v', '.wmv')
WEIGHT_EXTS = ('.pt', '.pth', '.onnx', '.engine')


# ----------------------------------------------------------------------
# 绘制工具
# ----------------------------------------------------------------------
def draw_overlay(frame: np.ndarray, state: str, sub_state: str,
                 detections: List[Dict[str, Any]]) -> np.ndarray:
    """画 state/sub_state 文字 (左上) + bbox (绿色 conf>0.5, 黄色 conf<=0.5)."""
    vis = frame.copy()
    cv2.putText(vis, f"state: {state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"sub : {sub_state}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det['bbox']]
        conf = float(det['conf'])
        color = (0, 255, 0) if conf > CONF_HI else (0, 200, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        ty = max(y1 - 6, 15)
        cv2.putText(vis, f"{conf:.2f}", (x1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return vis


def resize_keep_ar(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """等比缩放至 max_w x max_h 内 (不放大)."""
    h, w = frame.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
class CoilTipVizGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("钢卷头尾检测 GUI")
        self.root.geometry("1280x800")
        self.root.minsize(1100, 700)

        # ---- 运行状态 ----
        self.video_paths: List[str] = []
        self.current_video_idx: int = -1
        self.running: bool = False
        self.worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # 主线程 ← 后台: 帧 + 状态 (dict 协议, type 字段区分)
        self.q_frame: "queue.Queue[dict]" = queue.Queue(maxsize=2)
        # 主线程 → 后台: 控制命令 (预留; 当前无命令需要, worker 由 stop_event 控制)
        self.q_cmd: "queue.Queue[dict]" = queue.Queue()

        # 段导航
        self.change_segments: List[int] = []  # 已发现的 change 段起始帧 (append-only)
        self.current_segment_idx: int = -1     # 当前正在哪一段 (-1 = 未进入任何段)
        self.target_segment_idx: int = -1     # 后台 worker 要 seek 到的段索引 (-1 = 不跳转)

        # 模型 / 帧差
        self.detector: Optional[HyperYoloDetector] = None
        self.fd_proc: Optional[FrameDiffProcessor] = None
        # 用户选择的路径 (None = 用默认)
        self.weight_path: Optional[str] = None
        self.output_dir: str = CAPTURE_ROOT
        # conf 阈值 (用户可在 GUI 实时调整, 默认 0.15 = v18.3 部署最佳)
        self.conf_thr: float = 0.15

        # PhotoImage 引用 (防 GC)
        self._photo_ref: Optional[ImageTk.PhotoImage] = None

        # 必须先 _build_ui() 再 _init_detector/_init_fd_proc:
        # _init_* 会 self.lbl_weight.config()/self.lbl_output.config(), label 在 _build_ui 创建
        self._build_ui()
        # 启动时尝试加载默认权重; 失败不阻塞, 让用户手动选
        self._init_detector(use_default=True)
        self._init_fd_proc()
        self.lbl_video.config(text="当前: -- (请先选文件夹)")

    # ---- 初始化 ----
    def _init_detector(self, use_default: bool = False):
        """加载模型. use_default=True 时用 HyperYoloDetector 默认权重候选; 否则用 self.weight_path."""
        try:
            if use_default or not self.weight_path:
                self.detector = HyperYoloDetector(conf=self.conf_thr)
            else:
                self.detector = HyperYoloDetector(model_path=self.weight_path, conf=self.conf_thr)
            self.detector.warmup((1024, 1024, 3))
            self.lbl_weight.config(
                text=f"权重: {os.path.basename(self.detector.model_path)}",
                fg="black")
        except Exception as exc:
            self.detector = None
            self.lbl_weight.config(text="权重: 未加载", fg="red")
            self._show_error_blocking(
                f"模型加载失败:\n{type(exc).__name__}: {exc}\n\n"
                "请点击「选择权重」选择 .pt 文件")

    def _init_fd_proc(self):
        try:
            self.fd_proc = FrameDiffProcessor(out_dir=self.output_dir)
            self.lbl_output.config(text=f"输出: {self.output_dir}", fg="black")
        except Exception as exc:
            self.fd_proc = None
            self.lbl_output.config(text="输出: 未配置", fg="red")
            self._show_error_blocking(f"帧差法初始化失败: {type(exc).__name__}: {exc}")

    def _show_error_blocking(self, msg: str):
        # 初始化阶段, root 还没 mainloop, 用 messagebox 阻塞弹错
        try:
            messagebox.showerror("初始化失败", msg)
        except Exception:
            print(f"[FATAL] {msg}", file=sys.stderr)

    # ---- UI 构建 ----
    def _build_ui(self):
        # 顶部工具栏
        top = tk.Frame(self.root, height=80, bd=1, relief="raised")
        top.pack(side="top", fill="x")
        top.pack_propagate(False)
        # 第一行: 文件选择 + 播放控制
        row1 = tk.Frame(top)
        row1.pack(side="top", fill="x", padx=4, pady=2)
        tk.Button(row1, text="选择权重 (.pt)", command=self.on_choose_weight,
                  bg="#FFE4B5").pack(side="left", padx=2)
        tk.Button(row1, text="选择视频文件", command=self.on_choose_video_files,
                  bg="#FFE4B5").pack(side="left", padx=2)
        tk.Button(row1, text="选择视频文件夹", command=self.on_choose_folder).pack(side="left", padx=2)
        tk.Button(row1, text="选择输出目录", command=self.on_choose_output_dir).pack(side="left", padx=2)
        self.btn_run = tk.Button(row1, text="▶ 开始", command=self.on_toggle_run,
                                 width=10, bg="#90EE90")
        self.btn_run.pack(side="left", padx=8)
        # conf 阈值调整 (实时改 self.detector.conf)
        tk.Label(row1, text="conf ≥").pack(side="left", padx=(8, 2))
        self.spn_conf = tk.Spinbox(
            row1, from_=0.01, to=0.95, increment=0.05, width=5,
            format="%.2f",
            command=self.on_conf_changed,
        )
        self.spn_conf.delete(0, "end"); self.spn_conf.insert(0, f"{self.conf_thr:.2f}")
        self.spn_conf.pack(side="left", padx=2)
        # 绑定键盘修改 (Spinbox command 仅在点箭头触发; 键盘回车需手动绑)
        self.spn_conf.bind("<Return>", lambda _e: self.on_conf_changed())
        self.spn_conf.bind("<FocusOut>", lambda _e: self.on_conf_changed())
        # 第二行: 当前状态显示
        row2 = tk.Frame(top)
        row2.pack(side="top", fill="x", padx=4, pady=2)
        self.lbl_weight = tk.Label(row2, text="权重: 加载中...", anchor="w",
                                   font=("", 9), fg="blue")
        self.lbl_weight.pack(side="left", padx=4)
        self.lbl_output = tk.Label(row2, text=f"输出: {CAPTURE_ROOT}", anchor="w",
                                   font=("", 9))
        self.lbl_output.pack(side="left", padx=4)
        self.lbl_video = tk.Label(row2, text="视频: 未选", anchor="w",
                                  font=("", 9, "bold"))
        self.lbl_video.pack(side="left", padx=4, fill="x", expand=True)

        # 主体 (left list / center image / right panel)
        body = tk.Frame(self.root)
        body.pack(side="top", fill="both", expand=True)

        # ---- 左侧: 视频列表 ----
        left = tk.Frame(body, bd=1, relief="sunken")
        left.pack(side="left", fill="y")
        tk.Label(left, text="视频列表", anchor="w").pack(side="top", fill="x")
        self.lst_videos = tk.Listbox(left, width=30)
        self.lst_videos.pack(side="left", fill="both", expand=True)
        sb1 = tk.Scrollbar(left, command=self.lst_videos.yview)
        sb1.pack(side="right", fill="y")
        self.lst_videos.config(yscrollcommand=sb1.set)
        self.lst_videos.bind("<<ListboxSelect>>", self.on_select_video)

        # ---- 中央: 视频帧显示 ----
        center = tk.Frame(body, bd=1, relief="sunken", bg="black")
        center.pack(side="left", fill="both", expand=True)
        self.lbl_image = tk.Label(center, bg="black", fg="white", text="(无视频)")
        self.lbl_image.pack(expand=True, fill="both")

        # ---- 右侧: 状态面板 ----
        right = tk.Frame(body, width=300, bd=1, relief="sunken")
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # 状态信息
        info = tk.Frame(right)
        info.pack(side="top", fill="x", padx=4, pady=4)
        tk.Label(info, text="状态", font=("", 10, "bold")).pack(anchor="w")
        self.lbl_state = tk.Label(info, text="state: --", anchor="w")
        self.lbl_state.pack(anchor="w")
        self.lbl_sub = tk.Label(info, text="sub : --", anchor="w")
        self.lbl_sub.pack(anchor="w")
        self.lbl_fps = tk.Label(info, text="FPS: --", anchor="w")
        self.lbl_fps.pack(anchor="w")
        self.lbl_err = tk.Label(info, text="", fg="red", anchor="w",
                                wraplength=280, justify="left")
        self.lbl_err.pack(anchor="w", fill="x")

        # 检测结果 (conf > CONF_HI)
        det_frame = tk.Frame(right)
        det_frame.pack(side="top", fill="both", expand=True, padx=4, pady=4)
        tk.Label(det_frame, text=f"检测结果 (conf > {CONF_HI})", anchor="w").pack(fill="x")
        self.lst_det = tk.Listbox(det_frame, height=8)
        self.lst_det.pack(side="left", fill="both", expand=True)
        sb2 = tk.Scrollbar(det_frame, command=self.lst_det.yview)
        sb2.pack(side="right", fill="y")
        self.lst_det.config(yscrollcommand=sb2.set)

        # 截图列表
        cap_frame = tk.Frame(right)
        cap_frame.pack(side="top", fill="both", expand=True, padx=4, pady=4)
        tk.Label(cap_frame, text="截图列表", anchor="w").pack(fill="x")
        self.lst_cap = tk.Listbox(cap_frame, height=8)
        self.lst_cap.pack(side="left", fill="both", expand=True)
        sb3 = tk.Scrollbar(cap_frame, command=self.lst_cap.yview)
        sb3.pack(side="right", fill="y")
        self.lst_cap.config(yscrollcommand=sb3.set)

        # 底部控制
        bottom = tk.Frame(self.root, height=30, bd=1, relief="raised")
        bottom.pack(side="bottom", fill="x")
        bottom.pack_propagate(False)
        self.btn_prev = tk.Button(bottom, text="上一段", command=self.on_prev_segment, width=10)
        self.btn_prev.pack(side="left", padx=6, pady=2)
        self.btn_next = tk.Button(bottom, text="下一段", command=self.on_next_segment, width=10)
        self.btn_next.pack(side="left", padx=6, pady=2)
        self.lbl_seg = tk.Label(bottom, text="段: --/--", anchor="w")
        self.lbl_seg.pack(side="left", padx=12)

        # 窗口关闭
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------
    # 显示辅助
    # ------------------------------------------------------------------
    def _set_error(self, msg: str):
        self.lbl_err.config(text=msg)

    def _set_video_label(self, name: str, frame_idx: int, total: int, proc_fps: float):
        if total > 0:
            pct = frame_idx * 100.0 / total
            self.lbl_video.config(
                text=f"当前: {name}   [{frame_idx}/{total}] {pct:.1f}%   proc-FPS={proc_fps:.1f}")
        else:
            self.lbl_video.config(
                text=f"当前: {name}   [{frame_idx}]   proc-FPS={proc_fps:.1f}")

    def _set_seg_label(self):
        if self.current_segment_idx >= 0:
            self.lbl_seg.config(text=f"段: {self.current_segment_idx + 1}/{len(self.change_segments)}")
        else:
            self.lbl_seg.config(text=f"段: --/{len(self.change_segments)}")

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------
    def on_choose_weight(self):
        """弹出文件对话框选 .pt 权重文件"""
        path = filedialog.askopenfilename(
            title="选择模型权重 (.pt / .onnx / .engine)",
            filetypes=[
                ("YOLO 权重", "*.pt *.pth"),
                ("ONNX 模型", "*.onnx"),
                ("TensorRT 引擎", "*.engine"),
                ("所有文件", "*.*"),
            ])
        if not path:
            return
        self.weight_path = path
        # 立即重新加载
        self.lbl_weight.config(text=f"权重: 加载中... {os.path.basename(path)}", fg="blue")
        self.root.update()
        self._init_detector(use_default=False)

    def on_conf_changed(self):
        """conf 阈值 Spinbox 修改回调 - 实时更新 self.detector.conf (下一帧推理即生效)"""
        try:
            new_val = float(self.spn_conf.get())
        except (ValueError, tk.TclError):
            # 用户输入非数字 → 还原
            self.spn_conf.delete(0, "end"); self.spn_conf.insert(0, f"{self.conf_thr:.2f}")
            return
        # 钳位到合法范围
        new_val = max(0.01, min(0.95, new_val))
        self.conf_thr = new_val
        # 写回 Spinbox (避免精度漂移, 如 0.30000000000004)
        self.spn_conf.delete(0, "end"); self.spn_conf.insert(0, f"{new_val:.2f}")
        # 实时改 detector.conf (下一次 self.detector.detect() 即用新阈值)
        if self.detector is not None:
            self.detector.conf = new_val
        self.lbl_state.config(text=f"state: -- conf≥{new_val:.2f}")
        # 1 秒后还原 state label (避免盖掉下一帧真实状态)
        self.root.after(1000, lambda: self.lbl_state.config(text="state: --"))

    def on_choose_video_files(self):
        """弹出文件对话框选 1 个或多个视频文件"""
        paths = filedialog.askopenfilenames(
            title="选择视频文件 (可多选)",
            filetypes=[
                ("视频文件", " ".join(f"*{e}" for e in VIDEO_EXTS)),
                ("所有文件", "*.*"),
            ])
        if not paths:
            return
        self.video_paths = list(paths)
        self.lst_videos.delete(0, tk.END)
        for p in self.video_paths:
            self.lst_videos.insert(tk.END, os.path.basename(p))
        self._set_error("")
        if len(self.video_paths) == 1:
            name = os.path.basename(self.video_paths[0])
            self.lbl_video.config(text=f"视频: {name}   (点击▶ 开始)")
            self.current_video_idx = 0
            self._reset_for_new_video()
        else:
            self.lbl_video.config(
                text=f"视频: 已选 {len(self.video_paths)} 个文件   (列表选一个再开始)")

    def on_choose_output_dir(self):
        """弹出文件夹对话框选截图输出目录"""
        folder = filedialog.askdirectory(title="选择截图输出目录", initialdir=self.output_dir)
        if not folder:
            return
        self.output_dir = folder
        # 重建帧差实例指向新目录
        self._init_fd_proc()

    def on_choose_folder(self):
        folder = filedialog.askdirectory(title="选择视频文件夹")
        if not folder:
            return
        files = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.endswith(VIDEO_EXTS) and os.path.isfile(os.path.join(folder, f))
        )
        self.video_paths = files
        self.lst_videos.delete(0, tk.END)
        for p in files:
            self.lst_videos.insert(tk.END, os.path.basename(p))
        if not files:
            self._set_error(f"未在该目录下找到视频: {folder}")
            self.lbl_video.config(text="视频: 未选 (目录无视频)")
        else:
            self._set_error("")
            self.lbl_video.config(text=f"视频: 已扫描 {len(files)} 个, 选一个再开始")

    def on_select_video(self, _evt=None):
        sel = self.lst_videos.curselection()
        if not sel:
            return
        # 若正在跑, 先停
        if self.running:
            self._stop_worker()
        self.current_video_idx = int(sel[0])
        # 切换视频: 清空 listbox + 重置帧差 + 重置段状态
        self._reset_for_new_video()
        name = os.path.splitext(os.path.basename(self.video_paths[self.current_video_idx]))[0]
        self.lbl_video.config(text=f"视频: {name}   (未开始)")

    def on_toggle_run(self):
        if self.current_video_idx < 0 or not self.video_paths:
            messagebox.showwarning("提示", "请先选择视频文件或文件夹")
            return
        if self.detector is None:
            messagebox.showerror("错误", "模型未加载，请点击「选择权重」加载 .pt 文件")
            return
        if self.fd_proc is None:
            messagebox.showerror("错误", "帧差法未初始化")
            return

        self.running = not self.running
        self.btn_run.config(text="暂停" if self.running else "开始")
        if self.running:
            self._start_worker()
        else:
            self._stop_worker()

    def on_prev_segment(self):
        if not self.change_segments:
            self._set_error("尚未发现 change 段")
            return
        if self.current_segment_idx <= 0:
            self._set_error("已是第一段")
            return
        self.target_segment_idx = self.current_segment_idx - 1
        if not self.running:
            self._start_worker()
        else:
            self._set_error(f"请求跳转到段 #{self.target_segment_idx}...")

    def on_next_segment(self):
        if not self.change_segments:
            self._set_error("尚未发现 change 段")
            return
        next_idx = max(0, self.current_segment_idx + 1) if self.current_segment_idx >= 0 else 0
        if next_idx >= len(self.change_segments):
            self._set_error("已是最后一段")
            return
        self.target_segment_idx = next_idx
        if not self.running:
            self._start_worker()
        else:
            self._set_error(f"请求跳转到段 #{self.target_segment_idx}...")

    def on_close(self):
        self.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=1.0)
        self.root.destroy()

    # ------------------------------------------------------------------
    # 控制
    # ------------------------------------------------------------------
    def _start_worker(self):
        self.stop_event.clear()
        self.q_frame = queue.Queue(maxsize=2)
        self.q_cmd = queue.Queue()
        self.running = True  # 必须设, 否则 poll_queue() 首行 if not self.running: return 直接退出
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
        self._set_error("")
        self.root.after(POLL_MS, self.poll_queue)

    def _stop_worker(self):
        self.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=1.5)
        self.running = False
        self.btn_run.config(text="开始")
        self.target_segment_idx = -1  # 取消未执行的跳转

    def _reset_for_new_video(self):
        """切换视频时: 清空 listbox + 重置帧差 + 重置段/帧号."""
        self.lst_det.delete(0, tk.END)
        self.lst_cap.delete(0, tk.END)
        self.change_segments.clear()
        self.current_segment_idx = -1
        self.target_segment_idx = -1
        self.lbl_state.config(text="state: --")
        self.lbl_sub.config(text="sub : --")
        self.lbl_fps.config(text="FPS: --")
        self._set_seg_label()
        self._set_error("")
        self.lbl_image.config(image="", text="(无视频)")
        self._photo_ref = None
        if self.fd_proc is not None:
            # 清空 detector/capture 内部状态 (set_video_name 留给 worker)
            try:
                self.fd_proc.det.reset()
                if self.fd_proc.cc is not None:
                    self.fd_proc.cc.reset()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 后台 worker
    # ------------------------------------------------------------------
    def _worker(self):
        """读帧 → fd.process → detector.detect → 推 q_frame (dict 协议)."""
        video_path = ""
        cap = None
        try:
            if self.current_video_idx < 0:
                raise RuntimeError("未选择视频")
            video_path = self.video_paths[self.current_video_idx]
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            self.fd_proc.set_video_name(video_name)

            self._cur_cap = open_video(video_path)
            if not self._cur_cap.isOpened():
                raise RuntimeError(f"无法打开视频: {video_path}")
            total = int(self._cur_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps_native = float(self._cur_cap.get(cv2.CAP_PROP_FPS) or 25.0)

            frame_idx = 0          # 当前已处理帧数 (worker 局部)
            in_change = False      # 是否在 CHANGE 段内
            ignore_change_until_stable = False  # seek 后, 直到下一次 STABLE 再记录段
            t_fps = time.time()
            n_fps = 0
            proc_fps = 0.0

            while not self.stop_event.is_set():
                # 1. 处理跳转 (会替换 self._cur_cap)
                new_idx = self._maybe_seek(video_path, frame_idx)
                if new_idx is not None:
                    frame_idx = new_idx
                    in_change = False
                    ignore_change_until_stable = True  # 跳过当前 seek 到的段, 直到 STABLE
                    t_fps = time.time()
                    n_fps = 0

                # 2. 读帧
                ret, frame = self._cur_cap.read()
                if not ret or frame is None:
                    self.q_frame.put({"type": "end"})
                    break

                # 3. 帧差
                try:
                    state, sub_state, saved = self.fd_proc.process_frame(frame, fps=fps_native)
                except Exception as exc:
                    state, sub_state, saved = "ERR", f"{type(exc).__name__}", None
                    self.q_frame.put({"type": "err", "msg": f"fd 失败: {exc}"})

                # 4. 段边界: STABLE → CHANGE
                if state == "CHANGE" and not in_change:
                    self.change_segments.append(frame_idx)
                    in_change = True
                    new_seg_idx = len(self.change_segments) - 1
                    # 仅在非 seek 抑制期记录为当前段
                    if not ignore_change_until_stable:
                        self.current_segment_idx = new_seg_idx
                        self.q_frame.put({
                            "type": "new_seg",
                            "idx": new_seg_idx,
                            "start_frame": frame_idx,
                        })
                    else:
                        self.q_frame.put({
                            "type": "new_seg",
                            "idx": new_seg_idx,
                            "start_frame": frame_idx,
                            "ignored": True,
                        })
                elif state == "STABLE" and in_change:
                    in_change = False
                    ignore_change_until_stable = False

                # 5. 检测
                try:
                    dets = self.detector.detect(frame)
                except Exception as exc:
                    dets = []
                    self.q_frame.put({"type": "err", "msg": f"detect 失败: {exc}"})

                # 6. fps
                frame_idx += 1
                n_fps += 1
                if frame_idx % 30 == 0:
                    now = time.time()
                    proc_fps = n_fps / max(now - t_fps, 1e-6)
                    t_fps = now
                    n_fps = 0

                # 7. 推送
                vis = draw_overlay(frame, state, sub_state, dets)
                self.q_frame.put({
                    "type": "frame",
                    "vis": vis,
                    "frame_idx": frame_idx,
                    "total": total,
                    "fps_native": fps_native,
                    "proc_fps": proc_fps,
                    "state": state,
                    "sub_state": sub_state,
                    "saved": saved,
                    "dets": dets,
                    "video_name": video_name,
                })

        except Exception as exc:
            self.q_frame.put({
                "type": "err",
                "msg": f"worker 异常: {type(exc).__name__}: {exc}",
            })
        finally:
            try:
                if hasattr(self, '_cur_cap') and self._cur_cap is not None:
                    self._cur_cap.release()
            except Exception:
                pass
            self.q_frame.put({"type": "done"})

    def _maybe_seek(self, video_path: str, frame_idx: int) -> Optional[int]:
        """若 target_segment_idx 设定, 重开 + skip 到对应段起始帧. 返回新 frame_idx.

        把新 cap 写到 self._cur_cap (worker 也从此读).
        """
        if self.target_segment_idx < 0 or self.target_segment_idx >= len(self.change_segments):
            return None
        target = self.change_segments[self.target_segment_idx]
        if frame_idx == target:
            # 已到位, 落到该段
            self.current_segment_idx = self.target_segment_idx
            self.target_segment_idx = -1
            return None
        # 重开 + skip
        try:
            if hasattr(self, '_cur_cap') and self._cur_cap is not None:
                self._cur_cap.release()
        except Exception:
            pass
        new_cap = open_video(video_path)
        if not new_cap.isOpened():
            raise RuntimeError(f"seek 后无法重新打开: {video_path}")
        skipped = 0
        while skipped < target:
            ret, _ = new_cap.read()
            if not ret:
                break
            skipped += 1
        # 重置 fd (从 INIT 开始重放)
        self.fd_proc.reset()
        self._cur_cap = new_cap  # worker 用 self._cur_cap.read()
        self.current_segment_idx = self.target_segment_idx
        self.target_segment_idx = -1
        self.q_frame.put({
            "type": "seek_done",
            "target_frame": target,
        })
        return target

    # ------------------------------------------------------------------
    # 主线程 poll
    # ------------------------------------------------------------------
    def poll_queue(self):
        """每 100ms 拉取队列最新消息, frame 类型只保留最新一帧."""
        if not self.running:
            return
        try:
            latest_frame = None
            others: List[dict] = []
            try:
                # 先把所有消息读出来, frame 只保留最后一个 (其余丢弃, 不积压)
                while True:
                    msg = self.q_frame.get_nowait()
                    if msg.get("type") == "frame":
                        latest_frame = msg
                    else:
                        others.append(msg)
            except queue.Empty:
                pass
            # 先处理非 frame (err/end/new_seg/seek_done/done)
            for m in others:
                self._handle_main(m)
            # 再处理 frame
            if latest_frame is not None:
                self._handle_main(latest_frame)
        finally:
            if self.running:
                self.root.after(POLL_MS, self.poll_queue)

    def _handle_main(self, msg: dict):
        t = msg.get("type")
        if t == "frame":
            vis = msg["vis"]
            # 等比缩放至 Label 实际尺寸 (尚未布局时取上限)
            cw = max(self.lbl_image.winfo_width(), 100)
            ch = max(self.lbl_image.winfo_height(), 100)
            cw = min(cw, IMG_MAX_W)
            ch = min(ch, IMG_MAX_H)
            vis_small = resize_keep_ar(vis, cw, ch)
            rgb = cv2.cvtColor(vis_small, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(image=img)
            self.lbl_image.config(image=photo, text="")
            self._photo_ref = photo  # 防 GC

            self.lbl_state.config(text=f"state: {msg['state']}")
            self.lbl_sub.config(text=f"sub : {msg['sub_state']}")
            self.lbl_fps.config(
                text=f"FPS: 原 {msg['fps_native']:.1f} / 处理 {msg['proc_fps']:.1f}")
            self._set_video_label(
                msg['video_name'], msg['frame_idx'], msg['total'], msg['proc_fps'])

            # 检测结果 (conf > CONF_HI)
            for det in msg['dets']:
                if det['conf'] > CONF_HI:
                    b = det['bbox']
                    self.lst_det.insert(
                        tk.END,
                        f"f{msg['frame_idx']:>6} c{det['conf']:.2f} "
                        f"[{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])}]"
                    )
                    # 限制 Listbox 行数, 避免内存膨胀
                    if self.lst_det.size() > 500:
                        self.lst_det.delete(0)

            # 截图
            if msg.get('saved'):
                self.lst_cap.insert(tk.END, msg['saved'])
                if self.lst_cap.size() > 500:
                    self.lst_cap.delete(0)

            self._set_seg_label()

        elif t == "new_seg":
            idx = msg.get("idx", -1)
            ignored = msg.get("ignored", False)
            if not ignored and idx >= 0:
                self.current_segment_idx = idx
            self._set_seg_label()

        elif t == "seek_done":
            self._set_error(f"已跳转到 frame {msg.get('target_frame', '?')}")

        elif t == "err":
            self._set_error(msg.get('msg', ''))

        elif t == "end":
            self._set_error("视频已读完")
            self._stop_worker()

        elif t == "done":
            # worker 退出信号: 仅当仍在 running 时停 (用户已主动暂停时不重复)
            if self.running:
                self._stop_worker()


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def main():
    root = tk.Tk()
    app = CoilTipVizGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()