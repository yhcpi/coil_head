"""coil_tip_viz_gui.py - 钢卷头尾检测可视化主 GUI 程序

v2.0 - TRAE-style redesign, 2026-07-20

依赖模块 (已实现, 直接 import):
    frame_diff_wrapper.FrameDiffProcessor  (同目录)
    hyper_inference.HyperYoloDetector       (同目录)
    pyav_reader.open_video                  (/home/pi/projects/mm/帧差法/)

UI 布局 (ttkbootstrap cosmo, 1280x800):
    顶部导航 48px:  品牌 | 单一 tab "检测工作台" | 推理阈值 + 设置
    主体 12px margin 三列 (280 / 弹性 / 280):
        左:   资源 (4 按钮 + 权重/输出 meta) + 视频列表
        中:   卡片 [header 44 | 视频视口 | progress 4 | controls 56]
        右:   运行状态卡 + 检测卡 + 截图卡

线程模型 (不变):
    - 主线程: tkinter event loop
    - 后台 1 个 worker 线程 (daemon)
    - 主线程每 100ms 用 root.after(POLL_MS, poll_queue) 消费队列

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

import ttkbootstrap as ttk
from ttkbootstrap.constants import *

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
# 常量 (v1.0 保持不变)
# ----------------------------------------------------------------------
CAPTURE_ROOT = '/home/pi/projects/hyperyolo/runs/captures_gui'
CONF_HI = 0.5  # bbox 颜色分界: > CONF_HI 绿色, 否则黄色
POLL_MS = 100  # 主线程 poll queue 间隔
IMG_MAX_W, IMG_MAX_H = 880, 680  # 中央 Label 显示上限 (留出 toolbar/status 余量)
VIDEO_EXTS = ('.mp4', '.MP4', '.avi', '.AVI', '.mkv', '.MOV',
              '.mov', '.flv', '.ts', '.m4v', '.wmv')
WEIGHT_EXTS = ('.pt', '.pth', '.onnx', '.engine')


# ----------------------------------------------------------------------
# 设计 token (v2.0)
# ----------------------------------------------------------------------
COLOR_APP_BG        = "#F5F6F8"
COLOR_SIDEBAR_BG    = "#FAFBFC"
COLOR_CARD_BG       = "#FFFFFF"
COLOR_HOVER_BG      = "#F4F2FB"
COLOR_SELECTED_BG   = "#EEE9FF"
COLOR_VIDEO_BG      = "#17191F"
COLOR_BORDER        = "#E4E7EC"
COLOR_TEXT_PRIMARY  = "#20232A"
COLOR_TEXT_SECONDARY = "#596170"
COLOR_TEXT_MUTED    = "#8C95A3"
COLOR_ACCENT        = "#6B4EE6"
COLOR_ACCENT_SOFT   = "#F0ECFF"
COLOR_SUCCESS       = "#2E9663"
COLOR_SUCCESS_SOFT  = "#EAF7F0"
COLOR_WARNING       = "#D68A16"
COLOR_WARNING_SOFT  = "#FFF6E5"
COLOR_ERROR         = "#D14343"
COLOR_ERROR_SOFT    = "#FFF0F0"
COLOR_INFO          = "#3478F6"
COLOR_INFO_SOFT     = "#EDF4FF"

FONT_FAMILY      = "Segoe UI"
FONT_FAMILY_MONO = "Cascadia Mono"

# 优先字体 (中文+英文) — 在 Windows 上自动挑选第一个可用的
WIN_FONT_CANDIDATES      = ("Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI",
                            "PingFang SC", "Hiragino Sans GB", "SimSun", "Noto Sans CJK SC")
WIN_FONT_MONO_CANDIDATES = ("Cascadia Mono", "Cascadia Code", "Consolas",
                            "Source Code Pro", "JetBrains Mono", "Courier New")


def _pick_win_font(candidates: tuple, fallback: str) -> str:
    """从候选列表里挑当前 Tk 可见的第一个字体。

    没装任何 CJK 字体时回退到 fallback (Segoe UI / Cascadia Mono / TkDefaultFont)。
    Windows 10/11 自带 Microsoft YaHei UI + Cascadia Mono，所以实际部署一定能拿到。
    """
    try:
        import tkinter.font as tkfont
        # 必须在 root 存在后才能列出 family；本模块顶层未创建 root，所以
        # 此函数被首次调用时（_build_ui 创建 root 之前）会拿到空集合，
        # 此时直接返回 fallback。等真正渲染时 Tk 会自动通过系统 fallback 处理。
        available = set(tkfont.families())
    except Exception:
        return fallback
    for c in candidates:
        if c in available:
            return c
    return fallback


# 模块加载时执行一次（GUI 启动后此值仍可被 _build_ui 内的 pick 覆盖，
# 但 99% 情况下 _pick_win_font 在 root 创建后调用，结果就是 YaHei UI）
FONT_FAMILY      = _pick_win_font(WIN_FONT_CANDIDATES,      FONT_FAMILY)
FONT_FAMILY_MONO = _pick_win_font(WIN_FONT_MONO_CANDIDATES, FONT_FAMILY_MONO)
print(f"[coil_tip_viz_gui] FONT_FAMILY      = {FONT_FAMILY}")
print(f"[coil_tip_viz_gui] FONT_FAMILY_MONO = {FONT_FAMILY_MONO}")

# state -> 颜色映射 (用于徽章圆点)
STATE_COLORS = {
    "STABLE": COLOR_SUCCESS,
    "CHANGE": COLOR_WARNING,
    "ERR":    COLOR_ERROR,
    "INIT":   COLOR_INFO,
}


# ----------------------------------------------------------------------
# 绘制工具 (OpenCV BGR 标注色保持原样, 不替换为 UI 主题色)
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
# Tooltip 工具 (用于长路径悬停展示)
# ----------------------------------------------------------------------
class _Tooltip:
    """极简 Tooltip: 鼠标进入 widget 显示全文本."""
    def __init__(self, widget: tk.Widget, get_text):
        self.widget = widget
        self.get_text = get_text
        self.tip: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _evt=None):
        text = self.get_text()
        if not text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(tw, text=text, bg="#20232A", fg="#FFFFFF",
                       font=(FONT_FAMILY, 9), padx=8, pady=4,
                       justify="left", wraplength=420)
        lbl.pack()

    def _hide(self, _evt=None):
        if self.tip is not None:
            try:
                self.tip.destroy()
            except Exception:
                pass
            self.tip = None


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
class CoilTipVizGUI:
    def __init__(self, root: "ttk.Window"):
        global FONT_FAMILY, FONT_FAMILY_MONO
        self.root = root
        self.root.title("钢卷头尾检测 GUI")
        self.root.geometry("1280x800")
        self.root.minsize(1100, 700)

        # ---- 重新解析字体 (root 创建后 tkfont.families() 可用) ----
        FONT_FAMILY      = _pick_win_font(WIN_FONT_CANDIDATES,      FONT_FAMILY)
        FONT_FAMILY_MONO = _pick_win_font(WIN_FONT_MONO_CANDIDATES, FONT_FAMILY_MONO)

        # ---- 运行状态 (全部保留) ----
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
        # 当前 state key (用于徽章配色; 与 lbl_state 文本解耦, 便于样式更新)
        self._current_state_key: str = "INIT"

        # 必须先 _build_ui() 再 _init_detector/_init_fd_proc:
        # _init_* 会 self.lbl_weight.config()/self.lbl_output.config(), label 在 _build_ui 创建
        self._build_ui()
        # 启动时尝试加载默认权重; 失败不阻塞, 让用户手动选
        self._init_detector(use_default=True)
        self._init_fd_proc()
        self.lbl_video.config(text="当前: -- (请先选视频)")

    # ------------------------------------------------------------------
    # ttkbootstrap 样式集中定义
    # ------------------------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style()

        # ---- 容器 ----
        style.configure("App.TFrame",      background=COLOR_APP_BG)
        style.configure("TopNav.TFrame",   background=COLOR_CARD_BG)
        style.configure("Sidebar.TFrame",  background=COLOR_SIDEBAR_BG)
        style.configure("Card.TFrame",     background=COLOR_CARD_BG)
        style.configure("ControlBar.TFrame", background=COLOR_CARD_BG)
        style.configure("Video.TFrame",    background=COLOR_VIDEO_BG)
        style.configure("Header.TFrame",   background=COLOR_CARD_BG)

        # ---- 通用 Label ----
        style.configure("TLabel",
                        background=COLOR_CARD_BG,
                        foreground=COLOR_TEXT_PRIMARY,
                        font=(FONT_FAMILY, 11))

        # 顶部导航
        style.configure("Brand.TLabel",
                        background=COLOR_CARD_BG,
                        foreground=COLOR_TEXT_PRIMARY,
                        font=(FONT_FAMILY, 18, "bold"))
        style.configure("ActiveTab.TLabel",
                        background=COLOR_CARD_BG,
                        foreground=COLOR_ACCENT,
                        font=(FONT_FAMILY, 11, "bold"))

        # 通用 caption / section / meta
        style.configure("Caption.TLabel",
                        background=COLOR_CARD_BG,
                        foreground=COLOR_TEXT_MUTED,
                        font=(FONT_FAMILY, 9))
        style.configure("SectionTitle.TLabel",
                        background=COLOR_CARD_BG,
                        foreground=COLOR_TEXT_PRIMARY,
                        font=(FONT_FAMILY, 14, "bold"))
        style.configure("Meta.TLabel",
                        background=COLOR_CARD_BG,
                        foreground=COLOR_TEXT_MUTED,
                        font=(FONT_FAMILY, 9))
        style.configure("Secondary.TLabel",
                        background=COLOR_CARD_BG,
                        foreground=COLOR_TEXT_SECONDARY,
                        font=(FONT_FAMILY, 11))
        style.configure("Metric.TLabel",
                        background=COLOR_CARD_BG,
                        foreground=COLOR_TEXT_PRIMARY,
                        font=(FONT_FAMILY_MONO, 11))
        style.configure("Video.TLabel",
                        background=COLOR_VIDEO_BG,
                        foreground=COLOR_TEXT_MUTED,
                        font=(FONT_FAMILY, 11))

        # 侧栏变体 (不同背景)
        for base in ("SectionTitle.TLabel", "Caption.TLabel", "Meta.TLabel",
                     "Secondary.TLabel", "Brand.TLabel", "TLabel"):
            style.configure(f"Sidebar.{base}", background=COLOR_SIDEBAR_BG)

        # 状态徽章 (默认)
        style.configure("StatusBadge.Default.TLabel",
                        background=COLOR_ACCENT_SOFT,
                        foreground=COLOR_ACCENT,
                        font=(FONT_FAMILY, 11, "bold"),
                        padding=(10, 4))
        style.configure("StatusBadge.Success.TLabel",
                        background=COLOR_SUCCESS_SOFT,
                        foreground=COLOR_SUCCESS,
                        font=(FONT_FAMILY, 11, "bold"),
                        padding=(10, 4))
        style.configure("StatusBadge.Warning.TLabel",
                        background=COLOR_WARNING_SOFT,
                        foreground=COLOR_WARNING,
                        font=(FONT_FAMILY, 11, "bold"),
                        padding=(10, 4))
        style.configure("StatusBadge.Error.TLabel",
                        background=COLOR_ERROR_SOFT,
                        foreground=COLOR_ERROR,
                        font=(FONT_FAMILY, 11, "bold"),
                        padding=(10, 4))
        style.configure("StatusBadge.Info.TLabel",
                        background=COLOR_INFO_SOFT,
                        foreground=COLOR_INFO,
                        font=(FONT_FAMILY, 11, "bold"),
                        padding=(10, 4))

        # 段徽章
        style.configure("SegmentBadge.TLabel",
                        background=COLOR_ACCENT_SOFT,
                        foreground=COLOR_ACCENT,
                        font=(FONT_FAMILY, 11, "bold"),
                        padding=(10, 4))

        # 错误提示条
        style.configure("ErrorBanner.TLabel",
                        background=COLOR_ERROR_SOFT,
                        foreground=COLOR_ERROR,
                        font=(FONT_FAMILY, 10),
                        padding=(10, 6))

        # 卡片标题 (强制白底)
        style.configure("Card.SectionTitle.TLabel",
                        background=COLOR_CARD_BG,
                        foreground=COLOR_TEXT_PRIMARY,
                        font=(FONT_FAMILY, 14, "bold"))

    # ------------------------------------------------------------------
    # 初始化 (逻辑不变; lbl_weight/lbl_output 用 tk.Label 保留 fg=)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # UI 构建 (v2.0 - ttkbootstrap 布局)
    # ------------------------------------------------------------------
    def _build_ui(self):
        self._setup_styles()

        # ---- 根容器: 2 行 grid ----
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, minsize=48)   # 顶部导航 48px
        self.root.rowconfigure(1, minsize=1)    # 1px 分隔
        self.root.rowconfigure(2, weight=1)     # 主体占满

        # ============== 顶部导航 (48px) ==============
        topnav = ttk.Frame(self.root, style="TopNav.TFrame", height=48)
        topnav.grid(row=0, column=0, sticky="ew")
        topnav.pack_propagate(False)
        topnav.columnconfigure(0, minsize=280)
        topnav.columnconfigure(1, weight=1)
        topnav.columnconfigure(2, minsize=280)

        # 左侧品牌
        brand_box = ttk.Frame(topnav, style="TopNav.TFrame")
        brand_box.grid(row=0, column=0, sticky="w", padx=16)
        ttk.Label(brand_box, text="钢卷头尾检测",
                  style="Brand.TLabel").pack(side="left")
        ttk.Label(brand_box, text="可视化工作台",
                  style="Caption.TLabel").pack(side="left", padx=(8, 0))

        # 中间单一 tab
        tab_box = ttk.Frame(topnav, style="TopNav.TFrame")
        tab_box.grid(row=0, column=1)
        ttk.Label(tab_box, text="检测工作台",
                  style="ActiveTab.TLabel").pack(pady=14)

        # 右侧 conf + 设置
        right_box = ttk.Frame(topnav, style="TopNav.TFrame")
        right_box.grid(row=0, column=2, sticky="e", padx=16)
        ttk.Label(right_box, text="推理阈值",
                  style="Caption.TLabel").pack(side="left", padx=(0, 6))
        self.spn_conf = ttk.Spinbox(
            right_box, from_=0.01, to=0.95, increment=0.05, width=6,
            format="%.2f", bootstyle="primary",
            command=self.on_conf_changed,
        )
        self.spn_conf.delete(0, "end")
        self.spn_conf.insert(0, f"{self.conf_thr:.2f}")
        self.spn_conf.pack(side="left")
        self.spn_conf.bind("<Return>", lambda _e: self.on_conf_changed())
        self.spn_conf.bind("<FocusOut>", lambda _e: self.on_conf_changed())
        # 设置按钮 (聚焦 spn_conf, 不打开设置页)
        ttk.Button(right_box, text="设置", bootstyle="link",
                   command=self._focus_conf).pack(side="left", padx=(10, 0))

        # 1px 分隔 (Border 颜色)
        sep = tk.Frame(self.root, height=1, bg=COLOR_BORDER, bd=0)
        sep.grid(row=1, column=0, sticky="ew")

        # ============== 主体 (12px margin, 三列) ==============
        body = ttk.Frame(self.root, style="App.TFrame")
        body.grid(row=2, column=0, sticky="nsew", padx=12, pady=12)
        body.columnconfigure(0, minsize=280)
        body.columnconfigure(1, weight=1, minsize=480)
        body.columnconfigure(2, minsize=280)
        body.rowconfigure(0, weight=1)

        # ---- 左栏 ----
        self._build_left_sidebar(body)
        # ---- 中央卡 ----
        self._build_center_card(body)
        # ---- 右栏 ----
        self._build_right_sidebar(body)

        # 窗口关闭
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---- 左栏 ----
    def _build_left_sidebar(self, parent: ttk.Frame):
        left = tk.Frame(parent, bg=COLOR_SIDEBAR_BG, width=280,
                        highlightthickness=1, highlightbackground=COLOR_BORDER, bd=0)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        left.pack_propagate(False)
        left.columnconfigure(0, weight=1)

        # 资源区 (固定高度)
        res = tk.Frame(left, bg=COLOR_SIDEBAR_BG)
        res.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        res.columnconfigure(0, weight=1)

        ttk.Label(res, text="资源", style="Sidebar.SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8))

        # 4 个全宽按钮 + meta 标签
        ttk.Button(res, text="选择权重 (.pt)", bootstyle="secondary-outline",
                   command=self.on_choose_weight).grid(
            row=1, column=0, sticky="ew", pady=(0, 4))
        # lbl_weight 用原生 tk.Label 保留 fg= 能力
        self.lbl_weight = tk.Label(res, text="权重: 加载中...", anchor="w",
                                   bg=COLOR_SIDEBAR_BG, fg=COLOR_INFO,
                                   font=(FONT_FAMILY, 9), wraplength=252,
                                   justify="left")
        self.lbl_weight.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        ttk.Button(res, text="选择视频文件", bootstyle="primary-outline",
                   command=self.on_choose_video_files).grid(
            row=3, column=0, sticky="ew", pady=(0, 8))

        ttk.Button(res, text="选择视频文件夹", bootstyle="secondary-outline",
                   command=self.on_choose_folder).grid(
            row=4, column=0, sticky="ew", pady=(0, 8))

        ttk.Button(res, text="选择输出目录", bootstyle="secondary-outline",
                   command=self.on_choose_output_dir).grid(
            row=5, column=0, sticky="ew", pady=(0, 4))
        # lbl_output 用原生 tk.Label 保留 fg= 能力
        self.lbl_output = tk.Label(res, text=f"输出: {CAPTURE_ROOT}", anchor="w",
                                   bg=COLOR_SIDEBAR_BG, fg=COLOR_TEXT_PRIMARY,
                                   font=(FONT_FAMILY, 9), wraplength=252,
                                   justify="left")
        self.lbl_output.grid(row=6, column=0, sticky="ew")

        # 长路径 tooltip
        _Tooltip(self.lbl_weight, lambda: self._full_weight_text())
        _Tooltip(self.lbl_output, lambda: self._full_output_text())

        # 视频列表区 (占满剩余高度)
        list_section = tk.Frame(left, bg=COLOR_SIDEBAR_BG)
        list_section.grid(row=1, column=0, sticky="nsew", padx=12, pady=(4, 12))
        list_section.columnconfigure(0, weight=1)
        list_section.rowconfigure(1, weight=1)
        left.rowconfigure(1, weight=1)

        # 标题行 + 计数
        title_row = tk.Frame(list_section, bg=COLOR_SIDEBAR_BG)
        title_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        title_row.columnconfigure(0, weight=1)
        ttk.Label(title_row, text="视频列表",
                  style="Sidebar.SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.lbl_video_count = ttk.Label(title_row, text="0",
                                         style="Sidebar.Caption.TLabel")
        self.lbl_video_count.grid(row=0, column=1, sticky="e")

        # 列表 + 滚动条 (保留原生 Listbox)
        listbox_frame = tk.Frame(list_section, bg=COLOR_SIDEBAR_BG)
        listbox_frame.grid(row=1, column=0, sticky="nsew")
        listbox_frame.columnconfigure(0, weight=1)
        listbox_frame.rowconfigure(0, weight=1)
        self.lst_videos = tk.Listbox(
            listbox_frame,
            bg=COLOR_CARD_BG, fg=COLOR_TEXT_PRIMARY,
            selectbackground=COLOR_SELECTED_BG, selectforeground=COLOR_ACCENT,
            highlightthickness=0, bd=0, relief="flat",
            font=(FONT_FAMILY, 11), activestyle="none", borderwidth=0,
        )
        self.lst_videos.grid(row=0, column=0, sticky="nsew")
        try:
            sb1 = ttk.Scrollbar(listbox_frame, bootstyle="secondary-round",
                                command=self.lst_videos.yview)
        except Exception:
            sb1 = ttk.Scrollbar(listbox_frame, bootstyle="secondary",
                                command=self.lst_videos.yview)
        sb1.grid(row=0, column=1, sticky="ns")
        self.lst_videos.config(yscrollcommand=sb1.set)
        self.lst_videos.bind("<<ListboxSelect>>", self.on_select_video)

    # ---- 中央卡 ----
    def _build_center_card(self, parent: ttk.Frame):
        # 外层用 tk.Frame 包裹, 提供 1px 边框 (ttk.Frame 边框跨平台不稳)
        card_outer = tk.Frame(parent, bg=COLOR_BORDER, bd=0)
        card_outer.grid(row=0, column=1, sticky="nsew", padx=6)
        card_outer.columnconfigure(0, weight=1)
        card_outer.rowconfigure(0, weight=1)

        card = ttk.Frame(card_outer, style="Card.TFrame")
        card.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        card.columnconfigure(0, weight=1)
        card.rowconfigure(1, weight=1)  # 视频视口占据剩余空间

        # ---- header 44px ----
        header = ttk.Frame(card, style="Card.TFrame", height=44)
        header.grid(row=0, column=0, sticky="ew")
        header.pack_propagate(False)
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)

        ttk.Label(header, text="实时画面",
                  style="Card.SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=16, pady=12)

        # 段徽章 (放在 header 右侧)
        self.lbl_seg = ttk.Label(header, text="段: --/--", style="SegmentBadge.TLabel")
        self.lbl_seg.grid(row=0, column=1, sticky="e", padx=16, pady=10)

        # 1px 分隔线 (Border 颜色, 在 header 下方)
        sep_h = tk.Frame(card, height=1, bg=COLOR_BORDER, bd=0)
        sep_h.grid(row=0, column=0, sticky="sew")  # 与 header 同 row, 贴底
        # ↑ 上面这一行会与 header 重叠; 改用独立 row 放分隔线
        # 调整: 用 after 重新布局
        sep_h.lower()  # 放到最底层

        # ---- 视频视口 ----
        viewport = ttk.Frame(card, style="Video.TFrame")
        viewport.grid(row=1, column=0, sticky="nsew")
        viewport.columnconfigure(0, weight=1)
        viewport.rowconfigure(0, weight=1)
        self.lbl_image = ttk.Label(viewport, style="Video.TLabel",
                                   anchor="center", text="尚未选择视频")
        self.lbl_image.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # ---- 只读进度条 (4px, 仅镜像 frame/total) ----
        self.progress_bar = ttk.Progressbar(
            card, bootstyle="primary", mode="determinate",
            maximum=100, value=0,
        )
        self.progress_bar.grid(row=2, column=0, sticky="ew")

        # ---- 底部控制条 (56px) ----
        controls = ttk.Frame(card, style="ControlBar.TFrame", height=56)
        controls.grid(row=3, column=0, sticky="ew")
        controls.pack_propagate(False)
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)

        # 居中按钮组
        btn_cluster = ttk.Frame(controls, style="ControlBar.TFrame")
        btn_cluster.grid(row=0, column=1)
        self.btn_prev = ttk.Button(btn_cluster, text="上一段",
                                   bootstyle="secondary-outline",
                                   command=self.on_prev_segment, width=10)
        self.btn_prev.pack(side="left", padx=(0, 8))
        self.btn_run = ttk.Button(btn_cluster, text="▶ 开始",
                                  bootstyle="primary",
                                  command=self.on_toggle_run, width=14)
        self.btn_run.pack(side="left", padx=8)
        self.btn_next = ttk.Button(btn_cluster, text="下一段",
                                   bootstyle="secondary-outline",
                                   command=self.on_next_segment, width=10)
        self.btn_next.pack(side="left", padx=(8, 0))

    # ---- 右栏 ----
    def _build_right_sidebar(self, parent: ttk.Frame):
        right = tk.Frame(parent, bg=COLOR_SIDEBAR_BG, width=280,
                         highlightthickness=1, highlightbackground=COLOR_BORDER, bd=0)
        right.grid(row=0, column=2, sticky="ns", padx=(12, 0))
        right.pack_propagate(False)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=0)  # 状态卡
        right.rowconfigure(1, weight=1)  # 检测卡 (占满剩余)
        right.rowconfigure(2, weight=1)  # 截图卡 (占满剩余)

        self._build_status_card(right)
        self._build_detection_card(right)
        self._build_capture_card(right)

    def _build_status_card(self, parent: tk.Frame):
        card_outer = tk.Frame(parent, bg=COLOR_BORDER, bd=0)
        card_outer.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
        card_outer.columnconfigure(0, weight=1)

        card = ttk.Frame(card_outer, style="Card.TFrame")
        card.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        card.columnconfigure(0, weight=1)

        # 内边距通过 inner_frame 提供
        inner = ttk.Frame(card, style="Card.TFrame")
        inner.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)

        ttk.Label(inner, text="运行状态",
                  style="Card.SectionTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        # 状态徽章行 (canvas dot + ttk.Label)
        badge_row = ttk.Frame(inner, style="Card.TFrame")
        badge_row.grid(row=1, column=0, columnspan=2, sticky="w")
        self.lbl_state_dot = tk.Canvas(badge_row, width=12, height=12,
                                       bg=COLOR_CARD_BG,
                                       highlightthickness=0, bd=0)
        self.lbl_state_dot.pack(side="left", padx=(2, 8), pady=2)
        self._draw_state_dot(COLOR_INFO)
        self.lbl_state = ttk.Label(badge_row, text="state: --",
                                   style="StatusBadge.Info.TLabel")
        self.lbl_state.pack(side="left")

        # KV 行: sub / FPS
        self.lbl_sub = ttk.Label(inner, text="sub : --",
                                 style="Secondary.TLabel")
        self.lbl_sub.grid(row=2, column=0, sticky="w", pady=(12, 0))
        self.lbl_fps = ttk.Label(inner, text="FPS: --",
                                 style="Metric.TLabel")
        self.lbl_fps.grid(row=2, column=1, sticky="e", pady=(12, 0))

        # 当前视频 / 帧进度
        ttk.Label(inner, text="当前视频 / 帧进度",
                  style="Caption.TLabel").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(12, 4))
        self.lbl_video = ttk.Label(inner, text="当前: -- (请先选视频)",
                                   style="Secondary.TLabel",
                                   wraplength=244, justify="left")
        self.lbl_video.grid(row=4, column=0, columnspan=2, sticky="ew")

        # 错误提示条 (用单独容器以便空文本时收起视觉占位)
        self.err_holder = ttk.Frame(inner, style="Card.TFrame")
        self.err_holder.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self.err_holder.columnconfigure(0, weight=1)
        self.lbl_err = ttk.Label(self.err_holder, text="",
                                 style="ErrorBanner.TLabel",
                                 wraplength=228, justify="left")
        # 初始空文本: 隐藏视觉占位
        self.err_holder.grid_remove()

    def _build_detection_card(self, parent: tk.Frame):
        card_outer = tk.Frame(parent, bg=COLOR_BORDER, bd=0)
        card_outer.grid(row=1, column=0, sticky="nsew", pady=(0, 12))
        card_outer.columnconfigure(0, weight=1)
        card_outer.rowconfigure(0, weight=1)

        card = ttk.Frame(card_outer, style="Card.TFrame")
        card.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        card.columnconfigure(0, weight=1)
        card.rowconfigure(1, weight=1)

        # 标题行: "高置信检测" + caption "> 0.50"
        title_row = ttk.Frame(card, style="Card.TFrame")
        title_row.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 8))
        title_row.columnconfigure(0, weight=1)
        ttk.Label(title_row, text="高置信检测",
                  style="Card.SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(title_row, text="> 0.50",
                  style="Caption.TLabel").grid(row=0, column=1, sticky="e")

        # 列表 (保留 tk.Listbox)
        list_frame = ttk.Frame(card, style="Card.TFrame")
        list_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.lst_det = tk.Listbox(
            list_frame,
            bg=COLOR_CARD_BG, fg=COLOR_TEXT_PRIMARY,
            selectbackground=COLOR_SELECTED_BG, selectforeground=COLOR_ACCENT,
            highlightthickness=0, bd=0, relief="flat",
            font=(FONT_FAMILY_MONO, 9), activestyle="none", borderwidth=0,
        )
        self.lst_det.grid(row=0, column=0, sticky="nsew")
        try:
            sb2 = ttk.Scrollbar(list_frame, bootstyle="secondary-round",
                                command=self.lst_det.yview)
        except Exception:
            sb2 = ttk.Scrollbar(list_frame, bootstyle="secondary",
                                command=self.lst_det.yview)
        sb2.grid(row=0, column=1, sticky="ns")
        self.lst_det.config(yscrollcommand=sb2.set)

    def _build_capture_card(self, parent: tk.Frame):
        card_outer = tk.Frame(parent, bg=COLOR_BORDER, bd=0)
        card_outer.grid(row=2, column=0, sticky="nsew")
        card_outer.columnconfigure(0, weight=1)
        card_outer.rowconfigure(0, weight=1)

        card = ttk.Frame(card_outer, style="Card.TFrame")
        card.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        card.columnconfigure(0, weight=1)
        card.rowconfigure(1, weight=1)

        ttk.Label(card, text="截图列表",
                  style="Card.SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=16, pady=(12, 8))

        list_frame = ttk.Frame(card, style="Card.TFrame")
        list_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.lst_cap = tk.Listbox(
            list_frame,
            bg=COLOR_CARD_BG, fg=COLOR_TEXT_PRIMARY,
            selectbackground=COLOR_SELECTED_BG, selectforeground=COLOR_ACCENT,
            highlightthickness=0, bd=0, relief="flat",
            font=(FONT_FAMILY, 9), activestyle="none", borderwidth=0,
        )
        self.lst_cap.grid(row=0, column=0, sticky="nsew")
        try:
            sb3 = ttk.Scrollbar(list_frame, bootstyle="secondary-round",
                                command=self.lst_cap.yview)
        except Exception:
            sb3 = ttk.Scrollbar(list_frame, bootstyle="secondary",
                                command=self.lst_cap.yview)
        sb3.grid(row=0, column=1, sticky="ns")
        self.lst_cap.config(yscrollcommand=sb3.set)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _draw_state_dot(self, color: str):
        """在 lbl_state_dot 上画一个 10px 实心圆."""
        self.lbl_state_dot.delete("all")
        # 12x12 canvas, 画一个 r=5 圆点居中
        self.lbl_state_dot.create_oval(1, 1, 11, 11, fill=color, outline=color)

    def _refresh_state_indicator(self):
        """根据 self._current_state_key 同步徽章颜色与样式."""
        key = self._current_state_key
        if key == "STABLE":
            style_name = "StatusBadge.Success.TLabel"
            color = COLOR_SUCCESS
        elif key == "CHANGE":
            style_name = "StatusBadge.Warning.TLabel"
            color = COLOR_WARNING
        elif key == "ERR":
            style_name = "StatusBadge.Error.TLabel"
            color = COLOR_ERROR
        else:
            style_name = "StatusBadge.Info.TLabel"
            color = COLOR_INFO
        try:
            self.lbl_state.configure(style=style_name)
        except Exception:
            pass
        self._draw_state_dot(color)

    def _focus_conf(self):
        """设置按钮: 仅聚焦 spn_conf, 不打开新设置页."""
        try:
            self.spn_conf.focus_set()
            self.spn_conf.select_range(0, "end")
        except Exception:
            pass

    def _full_weight_text(self) -> str:
        if self.weight_path:
            return self.weight_path
        if self.detector is not None and getattr(self.detector, "model_path", None):
            return str(self.detector.model_path)
        return ""

    def _full_output_text(self) -> str:
        return self.output_dir or ""

    # ------------------------------------------------------------------
    # 显示辅助
    # ------------------------------------------------------------------
    def _set_error(self, msg: str):
        self.lbl_err.config(text=msg)
        # 空文本时收起视觉占位 (提示条只占需时空间)
        if msg:
            try:
                self.err_holder.grid()
            except Exception:
                pass
        else:
            try:
                self.err_holder.grid_remove()
            except Exception:
                pass

    def _set_video_label(self, name: str, frame_idx: int, total: int, proc_fps: float):
        if total > 0:
            pct = frame_idx * 100.0 / total
            self.lbl_video.config(
                text=f"当前: {name}   [{frame_idx}/{total}] {pct:.1f}%   proc-FPS={proc_fps:.1f}")
            # 同步进度条
            try:
                self.progress_bar['value'] = max(0.0, min(100.0, pct))
            except Exception:
                pass
        else:
            self.lbl_video.config(
                text=f"当前: {name}   [{frame_idx}]   proc-FPS={proc_fps:.1f}")
            try:
                self.progress_bar['value'] = 0
            except Exception:
                pass

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
        # 临时状态: 用 INFO 配色
        self._current_state_key = "INIT"
        self._refresh_state_indicator()
        # 1 秒后还原 state label (避免盖掉下一帧真实状态)
        self.root.after(1000, lambda: self._restore_state_label())

    def _restore_state_label(self):
        """on_conf_changed 1 秒后还原 lbl_state 文案与徽章."""
        self.lbl_state.config(text="state: --")
        self._current_state_key = "INIT"
        self._refresh_state_indicator()

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
        # 同步计数
        try:
            self.lbl_video_count.config(text=str(len(self.video_paths)))
        except Exception:
            pass
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
        # 同步计数
        try:
            self.lbl_video_count.config(text=str(len(files)))
        except Exception:
            pass
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
        # 运行时切到 amber/outline
        try:
            if self.running:
                self.btn_run.configure(bootstyle="warning-outline")
            else:
                self.btn_run.configure(bootstyle="primary")
        except Exception:
            pass
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
        try:
            self.btn_run.configure(bootstyle="primary")
        except Exception:
            pass
        self.target_segment_idx = -1  # 取消未执行的跳转

    def _reset_for_new_video(self):
        """切换视频时: 清空 listbox + 重置帧差 + 重置段/帧号."""
        self.lst_det.delete(0, tk.END)
        self.lst_cap.delete(0, tk.END)
        self.change_segments.clear()
        self.current_segment_idx = -1
        self.target_segment_idx = -1
        self.lbl_state.config(text="state: --")
        self._current_state_key = "INIT"
        self._refresh_state_indicator()
        self.lbl_sub.config(text="sub : --")
        self.lbl_fps.config(text="FPS: --")
        self._set_seg_label()
        self._set_error("")
        self.lbl_image.config(image="", text="(无视频)")
        self._photo_ref = None
        # 进度条归零
        try:
            self.progress_bar['value'] = 0
        except Exception:
            pass
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
            # 同步状态徽章颜色 (使用 message 中的 state)
            self._current_state_key = msg['state']
            self._refresh_state_indicator()
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
    app = ttk.Window(themename="cosmo")
    CoilTipVizGUI(app)
    app.mainloop()


if __name__ == "__main__":
    main()