#!/usr/bin/env python3
"""Standalone paddle-tracking calibration and diagnostics tool.

Run directly:
    python calibration_app.py

Or as a module from the parent directory:
    python -m ColorWar.calibration_app

Keyboard controls (active in the main window):
  SPACE   — Toggle between CALIBRATION and DETECTION phase
  H       — Hold calibration (prevent auto-switch to detection)
  R       — Reset detector (re-calibrate from scratch)
  S       — Save current session report + tuning profile to disk
  L       — Load last saved tuning profile
  P       — Toggle HSV trackbar panel (sliders)
  M       — Toggle motion mask on/off
  F       — Freeze frame (pause camera for inspection)
  5-8     — Mark scenario pass (face-on, edge-on, fast flick, background)
  Q / ESC — Quit
"""

import sys
import os
import cv2
import numpy as np
import time
import json
import math
from collections import defaultdict
from pathlib import Path

# Ensure the parent directory is on sys.path so relative imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ColorWar.config import (
    WIDTH, HEIGHT, PADDLE_RADIUS,
    PADDLE_HSV_PRIORS, HSV_FLOOR_CYAN, HSV_FLOOR_PINK,
    HSV_BLUR_KSIZE,
    MIN_CONTOUR_AREA, CONTOUR_AREA_MAX,
    CONTOUR_SOLIDITY_MIN, CONTOUR_COMPACTNESS_MIN,
    TRACKING_ASPECT_RATIO_MAX, TRACKING_MAX_JUMP,
    TRACKING_REACQUIRE_MAX_JUMP, TRACKING_CONFIDENCE_MIN,
    CONFIDENCE_HOLD, CONFIDENCE_GRACE_FRAMES,
    CALIB_LOCK_FRAMES, CALIB_LOCK_DECAY, PINK_CALIB_SAT_MIN,
    MIN_CONTOUR_AREA_EDGEON, EDGE_ON_ASPECT_MAX,
    DETECT_COLOR_WEIGHT, DETECT_EDGE_WEIGHT,
    DETECT_SHAPE_WEIGHT, DETECT_PROXIMITY_WEIGHT,
    HSV_REFINE_HUE_TOL, HSV_REFINE_SV_TOL,
    HSV_ADAPTIVE_BLEND, HSV_ADAPTIVE_INTERVAL,
    DETECT_USE_MOTION_MASK,
    MOTION_MASK_MIN_OVERLAP,
)
from ColorWar.tracking import PaddleTracker, _contour_quality, _passes_quality_strict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPORT_DIR = Path(__file__).resolve().parent / "calibration_reports"
_PROFILE_PATH = Path(__file__).resolve().parent / "calibration_profile.json"

# ---------------------------------------------------------------------------
# Colour constants for drawing (BGR)
# ---------------------------------------------------------------------------
COL_CYAN = (255, 255, 0)
COL_PINK = (180, 0, 255)
COL_GREEN = (0, 255, 0)
COL_YELLOW = (0, 255, 255)
COL_RED = (0, 0, 255)
COL_WHITE = (255, 255, 255)
COL_GRAY = (140, 140, 140)
COL_DARK = (40, 40, 40)
COL_ORANGE = (0, 140, 255)


# ===================================================================
# Session logger — collects per-frame metrics
# ===================================================================
class SessionLogger:
    def __init__(self):
        self.frames: list[dict] = []
        self.start_time = time.time()

    def log_frame(self, frame_idx: int, p1: dict, p2: dict):
        self.frames.append({
            "frame": frame_idx,
            "t": round(time.time() - self.start_time, 4),
            "p1": p1, "p2": p2,
        })

    def aggregate(self) -> dict:
        n = len(self.frames)
        if n == 0:
            return {"total_frames": 0}

        def _agg(key):
            color = [f[key]["color"] for f in self.frames]
            edge = [f[key]["edge"] for f in self.frames]
            shape = [f[key]["shape"] for f in self.frames]
            fused = [f[key]["fused"] for f in self.frames]
            rejects = [f[key].get("reject", "") for f in self.frames]

            def _pct(vals, th):
                return round(sum(1 for v in vals if v >= th) / n * 100, 1)

            rc = defaultdict(int)
            for r in rejects:
                if r:
                    rc[r] += 1
            top_rej = sorted(rc.items(), key=lambda x: -x[1])[:5]

            return {
                "color_valid_pct": _pct(color, 0.1),
                "edge_valid_pct": _pct(edge, 0.1),
                "shape_valid_pct": _pct(shape, 0.1),
                "fused_above_trust_pct": _pct(fused, TRACKING_CONFIDENCE_MIN),
                "avg_color": round(float(np.mean(color)), 3),
                "avg_edge": round(float(np.mean(edge)), 3),
                "avg_shape": round(float(np.mean(shape)), 3),
                "avg_fused": round(float(np.mean(fused)), 3),
                "top_reject_reasons": top_rej,
            }

        return {
            "total_frames": n,
            "duration_s": round(time.time() - self.start_time, 2),
            "p1": _agg("p1"), "p2": _agg("p2"),
        }

    def save(self, filepath: Path):
        report = self.aggregate()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return filepath


# ===================================================================
# Profile persistence
# ===================================================================
def save_profile(tracker: PaddleTracker, path: Path = _PROFILE_PATH):
    data = {}
    for label, is_p1 in [("cyan", True), ("pink", False)]:
        st = tracker._state[is_p1]
        data[label] = {
            "h_min": st.h_min, "h_max": st.h_max,
            "s_min": st.s_min, "s_max": st.s_max,
            "v_min": st.v_min, "v_max": st.v_max,
        }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_profile(tracker: PaddleTracker, path: Path = _PROFILE_PATH) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "r") as f:
            data = json.load(f)
        for label, is_p1 in [("cyan", True), ("pink", False)]:
            if label not in data:
                continue
            st = tracker._state[is_p1]
            for k in ("h_min", "h_max", "s_min", "s_max", "v_min", "v_max"):
                if k in data[label]:
                    setattr(st, k, int(data[label][k]))
            st.enforce_floors()
        return True
    except Exception:
        return False


# ===================================================================
# HSV Trackbar panel
# ===================================================================
_TRACKBAR_WIN = "HSV Tuning"


def _create_trackbar_window(tracker):
    cv2.namedWindow(_TRACKBAR_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_TRACKBAR_WIN, 500, 400)

    def _noop(_):
        pass

    for label, is_p1 in [("P1 Cyan", True), ("P2 Pink", False)]:
        st = tracker._state[is_p1]
        cv2.createTrackbar(f"{label} H min", _TRACKBAR_WIN, st.h_min, 179, _noop)
        cv2.createTrackbar(f"{label} H max", _TRACKBAR_WIN, st.h_max, 179, _noop)
        cv2.createTrackbar(f"{label} S min", _TRACKBAR_WIN, st.s_min, 255, _noop)
        cv2.createTrackbar(f"{label} S max", _TRACKBAR_WIN, st.s_max, 255, _noop)
        cv2.createTrackbar(f"{label} V min", _TRACKBAR_WIN, st.v_min, 255, _noop)
        cv2.createTrackbar(f"{label} V max", _TRACKBAR_WIN, st.v_max, 255, _noop)


def _read_trackbars(tracker):
    try:
        for label, is_p1 in [("P1 Cyan", True), ("P2 Pink", False)]:
            st = tracker._state[is_p1]
            st.h_min = cv2.getTrackbarPos(f"{label} H min", _TRACKBAR_WIN)
            st.h_max = cv2.getTrackbarPos(f"{label} H max", _TRACKBAR_WIN)
            st.s_min = cv2.getTrackbarPos(f"{label} S min", _TRACKBAR_WIN)
            st.s_max = cv2.getTrackbarPos(f"{label} S max", _TRACKBAR_WIN)
            st.v_min = cv2.getTrackbarPos(f"{label} V min", _TRACKBAR_WIN)
            st.v_max = cv2.getTrackbarPos(f"{label} V max", _TRACKBAR_WIN)
            st.enforce_floors()
    except cv2.error:
        pass


def _sync_trackbars(tracker):
    try:
        for label, is_p1 in [("P1 Cyan", True), ("P2 Pink", False)]:
            st = tracker._state[is_p1]
            cv2.setTrackbarPos(f"{label} H min", _TRACKBAR_WIN, st.h_min)
            cv2.setTrackbarPos(f"{label} H max", _TRACKBAR_WIN, st.h_max)
            cv2.setTrackbarPos(f"{label} S min", _TRACKBAR_WIN, st.s_min)
            cv2.setTrackbarPos(f"{label} S max", _TRACKBAR_WIN, st.s_max)
            cv2.setTrackbarPos(f"{label} V min", _TRACKBAR_WIN, st.v_min)
            cv2.setTrackbarPos(f"{label} V max", _TRACKBAR_WIN, st.v_max)
    except cv2.error:
        pass


# ===================================================================
# Drawing helpers
# ===================================================================
def _draw_all_contours(vis, hsv_blurred, tracker, is_p1, prev_x, prev_y):
    """Draw ALL HSV contours colour-coded (green=accepted, red=rejected)."""
    mask = tracker._get_mask(hsv_blurred, is_p1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reject_reasons = []

    for c in contours:
        area = cv2.contourArea(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw == 0 or bh == 0:
            cv2.drawContours(vis, [c], -1, (0, 0, 80), 1)
            continue
        aspect = max(bw / bh, bh / bw)
        cx_c = bx + bw // 2
        cy_c = by + bh // 2

        reason = ""
        if area < MIN_CONTOUR_AREA_EDGEON:
            reason = f"area<{MIN_CONTOUR_AREA_EDGEON}"
        elif area > CONTOUR_AREA_MAX:
            reason = f"area>{CONTOUR_AREA_MAX}"
        elif aspect > EDGE_ON_ASPECT_MAX:
            reason = f"asp>{EDGE_ON_ASPECT_MAX:.0f}"

        if reason:
            cv2.drawContours(vis, [c], -1, COL_RED, 1)
            cv2.putText(vis, reason, (cx_c + 5, cy_c + 5),
                         cv2.FONT_HERSHEY_PLAIN, 0.7, COL_RED, 1)
            reject_reasons.append(reason)
        else:
            # Annotate scores
            is_strong = _passes_quality_strict(c)
            tag = "OK" if is_strong else "EDGE"
            sol, comp = _contour_quality(c)
            cv2.drawContours(vis, [c], -1, COL_GREEN, 1)
            cv2.putText(vis, f"{tag} A:{int(area)} sol:{sol:.2f}",
                         (cx_c + 5, cy_c - 5),
                         cv2.FONT_HERSHEY_PLAIN, 0.7, COL_GREEN, 1)

    return reject_reasons


def _draw_conf_bar(vis, x, y, w, label, value, color, max_val=1.0):
    fill = int(w * min(value / max_val, 1.0))
    cv2.rectangle(vis, (x, y), (x + w, y + 14), COL_DARK, -1)
    if fill > 0:
        cv2.rectangle(vis, (x, y + 1), (x + fill, y + 13), color, -1)
    cv2.putText(vis, f"{label}:{value:.2f}", (x + 2, y + 11),
                 cv2.FONT_HERSHEY_PLAIN, 0.8, COL_WHITE, 1)


def _build_info_panel(tracker, is_p1, reject_reasons, frame_w):
    """Build a detailed status panel for one player."""
    st = tracker._state[is_p1]
    panel_h = 200
    panel = np.full((panel_h, frame_w, 3), 25, dtype=np.uint8)
    label = "P1 CYAN" if is_p1 else "P2 PINK"
    color = COL_CYAN if is_p1 else COL_PINK

    # Title
    cv2.putText(panel, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Lock / calibration progress
    if st.locked:
        cv2.putText(panel, "READY", (130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_GREEN, 2)
    else:
        pct = int(tracker.calibration_progress(is_p1) * 100)
        bar_x, bar_y, bar_w_full = 130, 8, 120
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w_full, bar_y + 16), COL_DARK, -1)
        fill = int(bar_w_full * pct / 100)
        if fill > 0:
            cv2.rectangle(panel, (bar_x, bar_y + 1), (bar_x + fill, bar_y + 15), COL_ORANGE, -1)
        cv2.putText(panel, f"Calib {pct}%", (bar_x + 3, bar_y + 13),
                     cv2.FONT_HERSHEY_PLAIN, 0.9, COL_WHITE, 1)

    # PARITY: Render mode badge — reflects actual hysteresis gate state
    if st._contour_held:
        if st._grace_counter < CONFIDENCE_GRACE_FRAMES and st.confidence < CONFIDENCE_HOLD:
            # Grace period active (contour held but confidence low)
            badge = f"RENDER: CONTOUR (grace {st._grace_counter})"
            cv2.putText(panel, badge, (270, 20),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_YELLOW, 2)
        else:
            cv2.putText(panel, "RENDER: CONTOUR", (290, 20),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_GREEN, 2)
    else:
        cv2.putText(panel, "RENDER: CIRCLE", (290, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_RED, 2)

    # Confidence score bars
    bar_w = 200
    _draw_conf_bar(panel, 10, 30, bar_w, "COLOR", st.color_score, COL_CYAN)
    _draw_conf_bar(panel, 10, 48, bar_w, "EDGE ", st.edge_score, COL_GREEN)
    _draw_conf_bar(panel, 10, 66, bar_w, "SHAPE", st.shape_score, COL_YELLOW)
    _draw_conf_bar(panel, 10, 84, bar_w, "PROX ", st.proximity_score, COL_ORANGE)
    _draw_conf_bar(panel, 10, 102, bar_w, "FUSED", st.confidence, COL_WHITE)

    # Trust threshold lines (enter and hold)
    th_enter_x = 10 + int(bar_w * TRACKING_CONFIDENCE_MIN)
    th_hold_x = 10 + int(bar_w * CONFIDENCE_HOLD)
    cv2.line(panel, (th_enter_x, 102), (th_enter_x, 116), COL_RED, 1)
    cv2.line(panel, (th_hold_x, 102), (th_hold_x, 116), COL_YELLOW, 1)
    cv2.putText(panel, "E", (th_enter_x - 3, 100),
                 cv2.FONT_HERSHEY_PLAIN, 0.6, COL_RED, 1)
    cv2.putText(panel, "H", (th_hold_x - 3, 100),
                 cv2.FONT_HERSHEY_PLAIN, 0.6, COL_YELLOW, 1)

    # Right column: HSV range + stats
    rx = 230
    cv2.putText(panel, f"H: {st.h_min}-{st.h_max}  S: {st.s_min}-{st.s_max}  V: {st.v_min}-{st.v_max}",
                 (rx, 42), cv2.FONT_HERSHEY_PLAIN, 0.9, COL_GRAY, 1)

    # Motion overlap
    mot_col = COL_GREEN if st.motion_score >= MOTION_MASK_MIN_OVERLAP else COL_RED
    cv2.putText(panel, f"Motion overlap: {st.motion_score:.2f}", (rx, 57),
                 cv2.FONT_HERSHEY_PLAIN, 0.9, mot_col, 1)

    # Miss count
    miss_col = COL_RED if st.miss_count > 5 else COL_GRAY
    cv2.putText(panel, f"Miss frames: {st.miss_count}", (rx, 72),
                 cv2.FONT_HERSHEY_PLAIN, 0.9, miss_col, 1)

    # Score weights reminder
    cv2.putText(panel, f"Weights C:{DETECT_COLOR_WEIGHT} E:{DETECT_EDGE_WEIGHT} "
                f"S:{DETECT_SHAPE_WEIGHT} P:{DETECT_PROXIMITY_WEIGHT}",
                 (rx, 87), cv2.FONT_HERSHEY_PLAIN, 0.75, COL_GRAY, 1)

    # ---- Failure guidance ----
    y_fail = 125
    if not st.locked and st.debug_lock_fail_reason:
        cv2.putText(panel, "CALIB FAIL: " + st.debug_lock_fail_reason[:55],
                     (10, y_fail), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_RED, 2)
        y_fail += 20

    if st.debug_reject_reason:
        cv2.putText(panel, "DETECT: " + st.debug_reject_reason[:55],
                     (10, y_fail), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_ORANGE, 1)
        y_fail += 18

    if reject_reasons:
        summary = ", ".join(set(reject_reasons))[:70]
        cv2.putText(panel, f"Contour rejects: {summary}",
                     (10, y_fail), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 100, 200), 1)
        y_fail += 15

    # ---- Actionable hint ----
    hint = ""
    if not st.locked:
        if not st.debug_lock_fail_reason:
            hint = "Hold paddle face-on to camera — need high saturation colour"
        elif "no HSV blobs" in st.debug_lock_fail_reason:
            hint = "No colour detected — check lighting and paddle colour"
        elif "low sat" in st.debug_lock_fail_reason:
            hint = "Paddle looks dim/washed out — improve lighting or angle"
        elif "area reject" in st.debug_lock_fail_reason:
            hint = "Blob too small or too large — adjust distance from camera"
        elif "shape reject" in st.debug_lock_fail_reason:
            hint = "Shape too irregular — show flat face of paddle"
        else:
            hint = "Keep paddle steady and visible to lock"
    elif not st._contour_held:
        # Use the specific fallback reason showing weakest cue
        fb = _get_fallback_reason(st)
        if fb:
            hint = fb
        else:
            hint = "Low confidence — multiple weak cues"

    if hint:
        cv2.putText(panel, hint, (10, y_fail + 5),
                     cv2.FONT_HERSHEY_PLAIN, 0.9, COL_YELLOW, 1)

    return panel


# ===================================================================
# Fallback reason (identifies weakest cue when confidence gate fails)
# ===================================================================
def _get_fallback_reason(st) -> str:
    """Concise reason why contour is not trusted (circle fallback).

    Accounts for hysteresis: if contour is currently held by grace frames,
    no fallback reason is shown.
    """
    # If hysteresis is holding the contour, it's not actually falling back
    if st._contour_held:
        return ""
    if st.confidence == 0.0:
        return "No paddle candidate found"
    if st.confidence >= TRACKING_CONFIDENCE_MIN:
        return ""
    cues = [
        ("COLOR", st.color_score, DETECT_COLOR_WEIGHT),
        ("EDGE", st.edge_score, DETECT_EDGE_WEIGHT),
        ("SHAPE", st.shape_score, DETECT_SHAPE_WEIGHT),
        ("PROX", st.proximity_score, DETECT_PROXIMITY_WEIGHT),
    ]
    weakest = min(cues, key=lambda c: c[1] * c[2])
    return (f"Weakest: {weakest[0]}={weakest[1]:.2f}  "
            f"fused {st.confidence:.2f} < enter {TRACKING_CONFIDENCE_MIN:.2f}")


# ===================================================================
# Game-equivalent render preview
# (PARITY: uses the exact same gating as main.py _compose_game_frame)
# ===================================================================
def _build_game_preview(frame, tracker, pos1, cnt1, pos2, cnt2):
    """Render paddles the same way gameplay does: contour when trusted,
    fallback circle when not.  Labels show render mode and confidence."""
    preview = frame.copy()

    for is_p1, pos, contour in [(True, pos1, cnt1), (False, pos2, cnt2)]:
        conf = tracker.get_confidence(is_p1)
        base_color = COL_CYAN if is_p1 else COL_PINK
        label = "P1" if is_p1 else "P2"

        # --- Dim colour when confidence is low (same as main.py) ---
        if conf >= TRACKING_CONFIDENCE_MIN:
            draw_col = base_color
        else:
            t = max(0.0, conf / TRACKING_CONFIDENCE_MIN)
            draw_col = tuple(int(c * t + 100 * (1 - t)) for c in base_color)

        px, py = int(pos[0]), int(pos[1])

        if contour is not None:
            # --- Trusted: draw actual contour (same as main.py) ---
            cv2.drawContours(preview, [contour], -1, draw_col, 3)
            render_tag = "CONTOUR"
            tag_col = COL_GREEN
        else:
            # --- Fallback: generic circle (same as main.py) ---
            cv2.circle(preview, (px, py), PADDLE_RADIUS, draw_col, 3)
            render_tag = "CIRCLE"
            tag_col = COL_RED

        # Render-mode label
        cv2.putText(preview, f"{label} {render_tag} conf:{conf:.2f}",
                     (max(5, px - 70), max(20, py - 40)),
                     cv2.FONT_HERSHEY_PLAIN, 1.0, tag_col, 1)

        # Fallback reason
        if contour is None:
            st = tracker._state[is_p1]
            reason = _get_fallback_reason(st)
            if reason:
                cv2.putText(preview, reason,
                             (max(5, px - 100), min(HEIGHT - 10, py + 50)),
                             cv2.FONT_HERSHEY_PLAIN, 0.8, COL_YELLOW, 1)

    # Title strip
    cv2.putText(preview, "GAME PREVIEW (same render as gameplay)",
                 (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_WHITE, 2)
    cv2.putText(preview, f"Enter:{TRACKING_CONFIDENCE_MIN:.2f}  Hold:{CONFIDENCE_HOLD:.2f}  "
                f"Grace:{CONFIDENCE_GRACE_FRAMES}f",
                 (10, 45), cv2.FONT_HERSHEY_PLAIN, 1.0, COL_GRAY, 1)

    return preview


# ===================================================================
# Scenario validation checklist
# ===================================================================
_SCENARIOS = [
    ("1. Face-on", "Hold paddle flat face to camera"),
    ("2. Edge-on", "Turn paddle 90deg — thin edge faces camera"),
    ("3. Fast flick", "Quickly swing paddle left-right"),
    ("4. Background", "Person walks behind with similar colour"),
]

_scenario_results: dict[int, str] = {}


def _draw_scenario_checklist(panel_w):
    h = 25 + len(_SCENARIOS) * 22
    panel = np.full((h, panel_w, 3), 30, dtype=np.uint8)
    cv2.putText(panel, "VALIDATION CHECKLIST (keys 5-8 = pass, shift+5-8 = fail)",
                 (10, 16), cv2.FONT_HERSHEY_PLAIN, 0.9, COL_WHITE, 1)
    for i, (name, desc) in enumerate(_SCENARIOS):
        y = 35 + i * 22
        result = _scenario_results.get(i, "")
        if result == "pass":
            col, mark = COL_GREEN, "[PASS]"
        elif result == "fail":
            col, mark = COL_RED, "[FAIL]"
        else:
            col, mark = COL_GRAY, "[    ]"
        cv2.putText(panel, f"{mark} {name}: {desc}", (10, y),
                     cv2.FONT_HERSHEY_PLAIN, 0.85, col, 1)
    return panel


# ===================================================================
# Main calibration app
# ===================================================================
def run_calibration():
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    tracker = PaddleTracker()

    phase = "calibration"
    hold_calibration = False
    frozen = False
    show_trackbars = False
    frame_idx = 0
    logger = SessionLogger()
    last_frame = None

    if load_profile(tracker):
        print("[CalibApp] Loaded saved profile from", _PROFILE_PATH)

    cv2.namedWindow("CalibApp", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CalibApp", WIDTH * 2, HEIGHT + 600)

    print("=" * 60)
    print("  Color War — Calibration & Diagnostics Tool")
    print("  Classical Detector: Color + Edge + Shape")
    print("=" * 60)
    print("  SPACE = toggle calibration/detection  H = hold calib")
    print("  R = reset   S = save report+profile   L = load profile")
    print("  P = HSV sliders   M = toggle motion mask")
    print("  F = freeze frame  5-8 = scenario pass (shift = fail)")
    print("  Q/ESC = quit")
    print("=" * 60)

    while True:
        if not frozen:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            last_frame = frame.copy()
        else:
            frame = last_frame.copy()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_blurred = tracker.blur_hsv(hsv)

        if show_trackbars:
            _read_trackbars(tracker)

        tracker.begin_frame(frame)

        # ---- Phase logic ----
        if phase == "calibration":
            tracker.calibrate_step(hsv, hsv_blurred)
            if tracker.both_locked and not hold_calibration:
                phase = "detection"
                print("[CalibApp] Both paddles locked — auto-switching to DETECTION")

        # PARITY: Always call track_paddle — exact same detection path as
        # gameplay in main.py — so the game preview is truthful.
        pos1, cnt1 = tracker.track_paddle(hsv_blurred, is_p1=True)
        pos2, cnt2 = tracker.track_paddle(hsv_blurred, is_p1=False)

        # Adaptive HSV only during detection phase (matches gameplay)
        if phase == "detection":
            tracker.adaptive_hsv_update(hsv, cnt1, is_p1=True)
            tracker.adaptive_hsv_update(hsv, cnt2, is_p1=False)

        tracker.end_frame()

        # ---- Build diagnostics visualisation ----
        vis_p1 = frame.copy()
        vis_p2 = frame.copy()

        st1 = tracker._state[True]
        st2 = tracker._state[False]

        # Previous / smoothed positions for drawing
        p1x, p1y = float(st1.smooth_pos[0]), float(st1.smooth_pos[1])
        p2x, p2y = float(st2.smooth_pos[0]), float(st2.smooth_pos[1])

        # Draw all contours
        rejects_p1 = _draw_all_contours(vis_p1, hsv_blurred, tracker, True, p1x, p1y)
        rejects_p2 = _draw_all_contours(vis_p2, hsv_blurred, tracker, False, p2x, p2y)

        # Draw detected position
        cv2.drawMarker(vis_p1, (int(p1x), int(p1y)),
                        COL_CYAN, cv2.MARKER_DIAMOND, 20, 2)
        cv2.drawMarker(vis_p2, (int(p2x), int(p2y)),
                        COL_PINK, cv2.MARKER_DIAMOND, 20, 2)

        # Draw proximity radius ring
        cv2.circle(vis_p1, (int(p1x), int(p1y)),
                    TRACKING_MAX_JUMP, (80, 80, 0), 1)
        cv2.circle(vis_p2, (int(p2x), int(p2y)),
                    TRACKING_MAX_JUMP, (80, 0, 80), 1)

        # ---- Mask + edge views ----
        mask_p1 = tracker._get_mask(hsv_blurred, True)
        mask_p2 = tracker._get_mask(hsv_blurred, False)
        mask_p1_bgr = cv2.cvtColor(mask_p1, cv2.COLOR_GRAY2BGR)
        mask_p2_bgr = cv2.cvtColor(mask_p2, cv2.COLOR_GRAY2BGR)
        mask_p1_bgr[:, :, 2] = (mask_p1 * 0.2).astype(np.uint8)
        mask_p2_bgr[:, :, 0] = (mask_p2 * 0.5).astype(np.uint8)
        mask_p2_bgr[:, :, 1] = (mask_p2 * 0.2).astype(np.uint8)

        # Edge map visualisation (shared for both players)
        edge_vis = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        if tracker._edge_map is not None:
            edge_vis[:, :, 1] = tracker._edge_map  # green channel

        # Motion mask visualisation
        motion_vis = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        if tracker._motion_mask is not None:
            motion_vis[:, :, 2] = tracker._motion_mask  # red channel

        # Resize for layout
        half_w = WIDTH
        vis_p1_s = cv2.resize(vis_p1, (half_w, HEIGHT))
        vis_p2_s = cv2.resize(vis_p2, (half_w, HEIGHT))
        mask_p1_s = cv2.resize(mask_p1_bgr, (half_w // 2, HEIGHT // 3))
        mask_p2_s = cv2.resize(mask_p2_bgr, (half_w // 2, HEIGHT // 3))
        edge_s = cv2.resize(edge_vis, (half_w // 2, HEIGHT // 3))
        motion_s = cv2.resize(motion_vis, (half_w // 2, HEIGHT // 3))

        # ---- Compose layout ----
        # Row 1: CV debug views (all contours, candidate markers)
        top_row = np.hstack([vis_p1_s, vis_p2_s])
        top_row = cv2.resize(top_row, (WIDTH * 2, HEIGHT))

        # Phase banner
        phase_col = COL_GREEN if phase == "detection" else COL_ORANGE
        phase_txt = f"PHASE: {phase.upper()}"
        if hold_calibration and phase == "calibration":
            phase_txt += " [HOLD]"
        cv2.putText(top_row, phase_txt, (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_col, 2)

        mot_txt = f"MotionMask:{'ON' if tracker._bg_sub is not None else 'OFF'}"
        cv2.putText(top_row, mot_txt, (10, 50),
                     cv2.FONT_HERSHEY_PLAIN, 1.0, COL_GRAY, 1)

        if frozen:
            cv2.putText(top_row, "FROZEN (F to unfreeze)", (WIDTH * 2 - 250, 25),
                         cv2.FONT_HERSHEY_PLAIN, 1.0, COL_RED, 1)
        cv2.putText(top_row, f"Frame: {frame_idx}", (WIDTH * 2 - 150, 50),
                     cv2.FONT_HERSHEY_PLAIN, 1.0, COL_GRAY, 1)

        # Row 2: GAME PREVIEW — exact same render as gameplay
        # (PARITY: contour when trusted, circle when not)
        game_preview = _build_game_preview(frame, tracker, pos1, cnt1, pos2, cnt2)
        game_row = cv2.resize(game_preview, (WIDTH * 2, int(HEIGHT * 0.55)))

        # Row 3: masks | edge | motion
        mid_left = np.hstack([mask_p1_s, mask_p2_s])
        mid_right = np.hstack([edge_s, motion_s])
        mask_row = np.hstack([mid_left, mid_right])
        mask_row = cv2.resize(mask_row, (WIDTH * 2, HEIGHT // 4))
        cv2.putText(mask_row, "P1 Cyan Mask", (5, 13),
                     cv2.FONT_HERSHEY_PLAIN, 0.8, COL_CYAN, 1)
        mw4 = WIDTH // 2
        cv2.putText(mask_row, "P2 Pink Mask", (mw4 + 5, 13),
                     cv2.FONT_HERSHEY_PLAIN, 0.8, COL_PINK, 1)
        cv2.putText(mask_row, "Edge Map", (mw4 * 2 + 5, 13),
                     cv2.FONT_HERSHEY_PLAIN, 0.8, COL_GREEN, 1)
        cv2.putText(mask_row, "Motion FG", (mw4 * 3 + 5, 13),
                     cv2.FONT_HERSHEY_PLAIN, 0.8, COL_RED, 1)

        # Row 4: Info panels
        panel_p1 = _build_info_panel(tracker, True, rejects_p1, WIDTH)
        panel_p2 = _build_info_panel(tracker, False, rejects_p2, WIDTH)
        bottom_row = np.hstack([panel_p1, panel_p2])

        # Row 5: Scenario checklist
        checklist = _draw_scenario_checklist(WIDTH * 2)

        # Stack all rows
        full = np.vstack([top_row, game_row, mask_row, bottom_row, checklist])
        full = cv2.resize(full, (WIDTH * 2, HEIGHT + 600))
        cv2.imshow("CalibApp", full)

        # ---- Session logging ----
        if phase == "detection":
            def _plog(is_p1):
                st = tracker._state[is_p1]
                return {
                    "color": round(st.color_score, 4),
                    "edge": round(st.edge_score, 4),
                    "shape": round(st.shape_score, 4),
                    "fused": round(st.confidence, 4),
                    "miss": st.miss_count,
                    "reject": st.debug_reject_reason,
                    "h_range": f"{st.h_min}-{st.h_max}",
                    "s_range": f"{st.s_min}-{st.s_max}",
                }
            logger.log_frame(frame_idx, _plog(True), _plog(False))

        frame_idx += 1

        # ---- Key handling ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            if phase == "calibration":
                if tracker.both_locked:
                    phase = "detection"
                    print("[CalibApp] Switched to DETECTION phase")
                else:
                    print("[CalibApp] Cannot switch — paddles not yet locked")
            else:
                phase = "calibration"
                print("[CalibApp] Switched to CALIBRATION phase")
        elif key == ord("h"):
            hold_calibration = not hold_calibration
            print(f"[CalibApp] Hold calibration: {'ON' if hold_calibration else 'OFF'}")
        elif key == ord("r"):
            tracker = PaddleTracker()
            phase = "calibration"
            logger = SessionLogger()
            frame_idx = 0
            _scenario_results.clear()
            print("[CalibApp] Detector RESET — back to calibration")
            if show_trackbars:
                _sync_trackbars(tracker)
        elif key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            report_path = _REPORT_DIR / f"report_{ts}.json"
            saved = logger.save(report_path)
            print(f"[CalibApp] Report saved: {saved}")
            prof = save_profile(tracker)
            print(f"[CalibApp] Profile saved: {prof}")
            _print_summary(logger.aggregate())
        elif key == ord("l"):
            if load_profile(tracker):
                print("[CalibApp] Profile loaded from", _PROFILE_PATH)
                if show_trackbars:
                    _sync_trackbars(tracker)
            else:
                print("[CalibApp] No profile found at", _PROFILE_PATH)
        elif key == ord("p"):
            show_trackbars = not show_trackbars
            if show_trackbars:
                _create_trackbar_window(tracker)
                print("[CalibApp] HSV trackbar panel OPENED")
            else:
                cv2.destroyWindow(_TRACKBAR_WIN)
                print("[CalibApp] HSV trackbar panel CLOSED")
        elif key == ord("m"):
            # Toggle motion mask
            if tracker._bg_sub is not None:
                tracker._bg_sub = None
                tracker._motion_mask = None
                print("[CalibApp] Motion mask DISABLED")
            else:
                tracker._bg_sub = cv2.createBackgroundSubtractorMOG2(
                    history=300, varThreshold=40, detectShadows=False,
                )
                print("[CalibApp] Motion mask ENABLED")
        elif key == ord("f"):
            frozen = not frozen
            print(f"[CalibApp] Frame {'FROZEN' if frozen else 'UNFROZEN'}")
        # Scenario marking
        elif key in (ord("5"), ord("%")):
            _scenario_results[0] = "fail" if key == ord("%") else "pass"
            print(f"[CalibApp] Scenario 1 (face-on): {_scenario_results[0]}")
        elif key in (ord("6"), ord("^")):
            _scenario_results[1] = "fail" if key == ord("^") else "pass"
            print(f"[CalibApp] Scenario 2 (edge-on): {_scenario_results[1]}")
        elif key in (ord("7"), ord("&")):
            _scenario_results[2] = "fail" if key == ord("&") else "pass"
            print(f"[CalibApp] Scenario 3 (fast flick): {_scenario_results[2]}")
        elif key in (ord("8"), ord("*")):
            _scenario_results[3] = "fail" if key == ord("*") else "pass"
            print(f"[CalibApp] Scenario 4 (background): {_scenario_results[3]}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    if logger.frames:
        ts = time.strftime("%Y%m%d_%H%M%S")
        report_path = _REPORT_DIR / f"report_{ts}_exit.json"
        logger.save(report_path)
        print(f"\n[CalibApp] Auto-saved exit report: {report_path}")
        _print_summary(logger.aggregate())

    if tracker.both_locked:
        save_profile(tracker)
        print(f"[CalibApp] Auto-saved final profile: {_PROFILE_PATH}")


def _print_summary(agg: dict):
    print("\n" + "=" * 60)
    print("  SESSION SUMMARY — Classical Detector")
    print("=" * 60)
    print(f"  Total frames: {agg.get('total_frames', 0)}")
    print(f"  Duration: {agg.get('duration_s', 0):.1f}s")

    for label in ("p1", "p2"):
        pa = agg.get(label, {})
        if not pa:
            continue
        name = "P1 CYAN" if label == "p1" else "P2 PINK"
        print(f"\n  --- {name} ---")
        print(f"  Fused trust (>{TRACKING_CONFIDENCE_MIN:.0%}): "
              f"{pa.get('fused_above_trust_pct', 0)}%")
        print(f"  Avg scores   Color={pa.get('avg_color', 0):.3f}  "
              f"Edge={pa.get('avg_edge', 0):.3f}  "
              f"Shape={pa.get('avg_shape', 0):.3f}  "
              f"Fused={pa.get('avg_fused', 0):.3f}")
        print(f"  Cue validity Color={pa.get('color_valid_pct', 0)}%  "
              f"Edge={pa.get('edge_valid_pct', 0)}%  "
              f"Shape={pa.get('shape_valid_pct', 0)}%")
        top_rej = pa.get("top_reject_reasons", [])
        if top_rej:
            print(f"  Top rejects:")
            for reason, count in top_rej:
                print(f"    - {reason}: {count}x")

    if _scenario_results:
        print(f"\n  --- SCENARIO RESULTS ---")
        for i, (name, _) in enumerate(_SCENARIOS):
            result = _scenario_results.get(i, "not tested")
            print(f"    {name}: {result.upper()}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_calibration()
