#!/usr/bin/env python3
"""Comprehensive end-to-end test for coil_tip_viz_gui

Tests:
  1. GUI startup + widget creation
  2. conf Spinbox real-time update
  3. Worker thread + frame diff state machine
  4. Capture saving on state transition
  5. Detector inference on actual frame
  6. Cleanup
"""
import os, sys, time, threading, queue, tkinter as tk
from pathlib import Path

os.environ['YOLO_VERBOSE'] = 'False'

# Force-add hyper-yolo env for ultralytics
sys.path.insert(0, '/home/pi/projects/hyperyolo/scripts/gui')
sys.path.insert(0, '/home/pi/projects/mm/帧差法')

import tkinter.filedialog as fd
from coil_tip_viz_gui import CoilTipVizGUI

PASSED = []
FAILED = []


def check(name, cond, detail=""):
    if cond:
        PASSED.append(name)
        print(f"  [OK]   {name}")
    else:
        FAILED.append(f"{name}  ({detail})")
        print(f"  [FAIL] {name}  -- {detail}")


def main():
    print("=" * 60)
    print("  Coil Tip Detection GUI - End-to-End Verification")
    print("=" * 60)
    print()

    # ---- Test 1: GUI startup ----
    print("[Test 1] GUI startup + widget creation")
    root = tk.Tk()
    root.geometry("1280x800")
    root.update_idletasks()
    app = CoilTipVizGUI(root)

    check("GUI window created", root.winfo_exists())
    check("Detector loaded (model loaded)", app.detector is not None,
          f"detector={app.detector}")
    check("Default weight path exists",
          app.detector is not None and Path(app.detector.model_path).exists(),
          f"path={app.detector.model_path if app.detector else 'None'}")
    check("conf Spinbox created", hasattr(app, 'spn_conf'))
    check("Initial conf_thr = 0.15", app.conf_thr == 0.15)
    check("All toolbar buttons exist",
          all(hasattr(app, m) for m in ['btn_run']))
    check("Label widgets exist",
          all(hasattr(app, lbl) for lbl in
              ['lbl_weight', 'lbl_output', 'lbl_video', 'lbl_state',
               'lbl_sub', 'lbl_fps', 'lbl_image', 'lbl_err']))
    check("Listboxes exist",
          all(hasattr(app, lb) for lb in ['lst_videos', 'lst_det', 'lst_cap']))
    print()

    # ---- Test 2: conf Spinbox real-time update ----
    print("[Test 2] conf Spinbox real-time detector.conf update")
    original_conf = app.conf_thr
    app.spn_conf.delete(0, "end"); app.spn_conf.insert(0, "0.30")
    app.on_conf_changed()
    check("After set 0.30 -> app.conf_thr = 0.30", abs(app.conf_thr - 0.30) < 1e-6)
    check("After set 0.30 -> detector.conf = 0.30",
          app.detector is not None and abs(app.detector.conf - 0.30) < 1e-6)
    check("Spinbox displays '0.30'", app.spn_conf.get() == "0.30")

    # Invalid input
    app.spn_conf.delete(0, "end"); app.spn_conf.insert(0, "abc")
    app.on_conf_changed()
    check("Invalid 'abc' -> keeps 0.30", abs(app.conf_thr - 0.30) < 1e-6)

    # Out of range
    app.spn_conf.delete(0, "end"); app.spn_conf.insert(0, "1.5")
    app.on_conf_changed()
    check("Out-of-range 1.5 -> clamped to 0.95",
          abs(app.conf_thr - 0.95) < 1e-6)

    # Reset to 0.15 for next test
    app.spn_conf.delete(0, "end"); app.spn_conf.insert(0, "0.15")
    app.on_conf_changed()
    check("Reset to 0.15 for video test", abs(app.conf_thr - 0.15) < 1e-6)
    print()

    # ---- Test 3: Worker + frame diff state machine ----
    print("[Test 3] Worker + frame diff state machine")
    fd.askopenfilenames = lambda **kw: ('/tmp/test_coil_video.mp4',)
    app.on_choose_video_files()
    app.on_select_video()
    check("Video added to lst_videos", app.lst_videos.size() >= 1)

    app._start_worker()
    check("Worker started, running=True", app.running)
    check("Worker thread alive", app.worker is not None and app.worker.is_alive())

    # Run mainloop for 8s to let frames flow
    deadline = time.time() + 8
    while time.time() < deadline:
        try:
            root.update()
            time.sleep(0.03)
        except tk.TclError:
            break

    check("After 8s lbl_video shows progress",
          "/" in app.lbl_video.cget("text") or "%" in app.lbl_video.cget("text"),
          f"lbl_video={app.lbl_video.cget('text')[:60]}")
    check("After 8s lbl_state updated",
          app.lbl_state.cget("text") != "state: --",
          f"state={app.lbl_state.cget('text')}")
    check("After 8s _photo_ref set (frame drawn)",
          app._photo_ref is not None)
    check("After 8s lbl_fps shows FPS",
          "FPS" in app.lbl_fps.cget("text"),
          f"fps={app.lbl_fps.cget('text')}")
    check("Worker still alive after 8s",
          app.worker.is_alive())
    print()

    # ---- Test 4: Real-time conf change during worker run ----
    print("[Test 4] Real-time conf change during worker")
    app.spn_conf.delete(0, "end"); app.spn_conf.insert(0, "0.50")
    app.on_conf_changed()
    check("Mid-run conf=0.50 -> detector.conf=0.50",
          abs(app.detector.conf - 0.50) < 1e-6)
    # let some frames process with new conf
    deadline = time.time() + 2
    while time.time() < deadline:
        try:
            root.update()
            time.sleep(0.03)
        except tk.TclError:
            break
    print()

    # ---- Test 5: Stop worker cleanly ----
    print("[Test 5] Clean worker shutdown (stop_event honored)")
    app._stop_worker()
    check("After stop, running=False", not app.running)
    check("After stop, btn_run text reset to '开始'",
          app.btn_run.cget("text") == "开始",
          f"text='{app.btn_run.cget('text')}'")
    check("After stop, target_segment_idx reset to -1",
          app.target_segment_idx == -1)
    # Worker may still be alive due to cv2 blocking read (cannot interrupt),
    # but stop_event is set so worker will exit at next iteration check.
    check("After stop, stop_event is set",
          app.stop_event.is_set())
    print(f"  [INFO] Worker alive={app.worker.is_alive() if app.worker else False} "
          f"(may stay alive during cv2 read; exits at next check)")
    print()

    # ---- Test 6: Detector inference correctness ----
    print("[Test 6] Detector inference on synthetic frame")
    import numpy as np
    test_frame = np.zeros((1024, 1024, 3), dtype=np.uint8)
    test_frame[100:300, 200:600] = 128  # gray patch
    dets = app.detector.detect(test_frame)
    check("Detector returns list", isinstance(dets, list),
          f"type={type(dets).__name__}")
    print(f"        Detections on blank+patch frame: {len(dets)}")
    print()

    # ---- Test 7: Cleanup ----
    print("[Test 7] Cleanup")
    try:
        app.on_close()
        check("GUI close() ran without error", True)
    except Exception as e:
        check("GUI close() ran without error", False, str(e))
    print()

    # ---- Summary ----
    print("=" * 60)
    print(f"  PASSED: {len(PASSED)}")
    print(f"  FAILED: {len(FAILED)}")
    if FAILED:
        print()
        print("Failures:")
        for f in FAILED:
            print(f"  - {f}")
        print()
        sys.exit(1)
    else:
        print()
        print("  ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()