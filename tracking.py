"""Per-frame classical paddle detector (color + edge + shape).

No deep-learning dependencies.  No temporal state machines (CSRT, optical-flow,
Kalman).  Every frame is a fresh detection pass that scores candidates using:

  1. **Colour score** — HSV in-range purity and saturation strength.
  2. **Edge score** — Canny edge density along the contour boundary.
  3. **Shape score** — solidity, compactness, area, aspect ratio.
  4. **Proximity bonus** — soft tie-breaker from previous-frame position.
  5. **Motion mask** — optional MOG2 background subtraction to reject static
     distractors (coloured walls, furniture).

Position is smoothed with an exponential moving average (EMA) to reduce jitter
while remaining responsive.  Confidence is purely per-frame and used downstream
for collision gating.
"""

import cv2
import numpy as np
import time as _time
import math

from .config import (
    WIDTH,
    HEIGHT,
    MIN_CONTOUR_AREA,
    TRACKING_REACQUIRE_MAX_JUMP,
    TRACKING_ASPECT_RATIO_MAX,
    PADDLE_HSV_PRIORS,
    HSV_FLOOR_CYAN,
    HSV_FLOOR_PINK,
    HSV_BLUR_KSIZE,
    HSV_REFINE_HUE_TOL,
    HSV_REFINE_SV_TOL,

    CALIB_LOCK_FRAMES,
    CALIB_LOCK_DECAY,
    PINK_CALIB_SAT_MIN,
    CONTOUR_SOLIDITY_MIN,
    CONTOUR_COMPACTNESS_MIN,
    CONTOUR_AREA_MAX,
    TRACKING_CONFIDENCE_MIN,
    MIN_CONTOUR_AREA_EDGEON,
    EDGE_ON_ASPECT_MAX,
    # Classical detector
    DETECT_COLOR_WEIGHT,
    DETECT_EDGE_WEIGHT,
    DETECT_SHAPE_WEIGHT,
    DETECT_PROXIMITY_WEIGHT,
    EDGE_CANNY_LOW,
    EDGE_CANNY_HIGH,
    EMA_ALPHA,

    # Hysteresis gate
    CONFIDENCE_HOLD,
    CONFIDENCE_GRACE_FRAMES,
    # Kalman filter
    KALMAN_PROCESS_NOISE,
    KALMAN_MEASUREMENT_NOISE,
    KALMAN_MISS_LIMIT,
    # CSRT tracker
    USE_CSRT_FALLBACK,
    CSRT_CONFIDENCE_MIN,
    CSRT_REINIT_INTERVAL,
    # LAB color
    DETECT_LAB_WEIGHT,
    LAB_DIST_THRESHOLD,
    # Contour smoothing
    CONTOUR_SMOOTH_FRAMES,
    CONTOUR_RESAMPLE_POINTS,
)
from collections import deque


# ---------------------------------------------------------------------------
# Contour quality helpers (unchanged — used by calibration_app too)
# ---------------------------------------------------------------------------
def _contour_quality(c: np.ndarray) -> tuple[float, float]:
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    perimeter = cv2.arcLength(c, True)
    compactness = (4.0 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
    return solidity, compactness


def _passes_quality_strict(c: np.ndarray) -> bool:
    """Strict quality check — full-face paddle view."""
    area = cv2.contourArea(c)
    if area < MIN_CONTOUR_AREA or area > CONTOUR_AREA_MAX:
        return False
    bx, by, bw, bh = cv2.boundingRect(c)
    if bw == 0 or bh == 0:
        return False
    aspect = max(bw / bh, bh / bw)
    if aspect > TRACKING_ASPECT_RATIO_MAX:
        return False
    solidity, compactness = _contour_quality(c)
    if solidity < CONTOUR_SOLIDITY_MIN:
        return False
    if compactness < CONTOUR_COMPACTNESS_MIN:
        return False
    return True


# ---------------------------------------------------------------------------
# Per-player detection state
# ---------------------------------------------------------------------------
class _PlayerDetectState:
    def __init__(self, initial_pos: np.ndarray, prior: dict, is_p1: bool):
        self.is_p1: bool = is_p1

        # HSV range (tuned during calibration, adapted during gameplay)
        self.h_min: int = prior["h_min"]
        self.h_max: int = prior["h_max"]
        self.s_min: int = prior["s_min"]
        self.s_max: int = prior["s_max"]
        self.v_min: int = prior["v_min"]
        self.v_max: int = prior["v_max"]

        # Calibration
        self.locked: bool = False
        self.lock_counter: int = 0



        # Position history (for proximity bonus + EMA smoothing)
        self.prev_pos: np.ndarray = initial_pos.copy()
        self.smooth_pos: np.ndarray = initial_pos.copy()

        # Miss tracking
        self.miss_count: int = 0
        self.last_seen_time: float = _time.time()

        # Per-cue scores (for diagnostics / calibration UI)
        self.color_score: float = 0.0
        self.edge_score: float = 0.0
        self.shape_score: float = 0.0
        self.proximity_score: float = 0.0

        self.confidence: float = 0.0   # fused

        # Hysteresis / grace state for contour persistence
        self._contour_held: bool = False
        self._grace_counter: int = 0
        self._last_contour: np.ndarray | None = None

        # Debug
        self.debug_reject_reason: str = ""
        self.debug_lock_fail_reason: str = ""

        # Calibration spatial consistency: anchor position during lock-up
        self.calib_anchor_pos: np.ndarray | None = None

        # Size continuity: last accepted contour area (for jump rejection)
        self.last_area: float = 0.0

        # --- Kalman filter (6-state: x, y, dx, dy, w, h) ---
        self.kalman: cv2.KalmanFilter = self._init_kalman(initial_pos)
        self.kalman_miss_count: int = 0
        self.kalman_pos: np.ndarray = initial_pos.copy()  # predicted position

        # --- CSRT visual tracker ---
        self.csrt: cv2.Tracker | None = None
        self.csrt_bbox: tuple | None = None
        self.csrt_good_count: int = 0  # counts successful detections for re-init

        # --- LAB reference color (set during calibration click) ---
        self.lab_ref: np.ndarray | None = None  # mean LAB of paddle pixels

        # --- Temporal contour smoothing ---
        self.contour_history: deque = deque(maxlen=CONTOUR_SMOOTH_FRAMES)

    @staticmethod
    def _init_kalman(initial_pos: np.ndarray) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(6, 4, 0)  # 6 state vars, 4 measurements
        # Transition matrix: constant-velocity model
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        # Measurement matrix: we observe x, y, w, h
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * KALMAN_PROCESS_NOISE
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE
        kf.errorCovPost = np.eye(6, dtype=np.float32)
        kf.statePost = np.array([
            initial_pos[0], initial_pos[1], 0, 0, 40, 40,
        ], dtype=np.float32)
        return kf

    def kalman_predict(self) -> np.ndarray:
        """Run Kalman predict step AND return predicted [x, y]."""
        pred = self.kalman.predict()
        self.kalman_pos = np.array([float(pred[0, 0]), float(pred[1, 0])], dtype=float)
        return self.kalman_pos

    def kalman_correct(self, cx: float, cy: float, w: float, h: float):
        """Feed measurement to Kalman."""
        measurement = np.array([cx, cy, w, h], dtype=np.float32)
        self.kalman.correct(measurement)
        self.kalman_miss_count = 0

    def kalman_mark_miss(self):
        self.kalman_miss_count += 1
        if self.kalman_miss_count >= KALMAN_MISS_LIMIT:
            # Reset Kalman to last known good position
            self.kalman = self._init_kalman(self.smooth_pos)
            self.kalman_miss_count = 0

    @property
    def hsv_floor(self) -> dict:
        return HSV_FLOOR_CYAN if self.is_p1 else HSV_FLOOR_PINK

    def enforce_floors(self):
        floor = self.hsv_floor
        if "s_min" in floor:
            self.s_min = max(self.s_min, floor["s_min"])
        if "v_min" in floor:
            self.v_min = max(self.v_min, floor["v_min"])
        if "h_min" in floor:
            self.h_min = max(self.h_min, floor["h_min"])
        if "h_max" in floor:
            self.h_max = min(self.h_max, floor["h_max"])


# ---------------------------------------------------------------------------
# Morphology kernels
# ---------------------------------------------------------------------------
_KERN_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_KERN_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
_KERN_OPEN_THIN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_KERN_CLOSE_THIN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_KERN_COLOR_INNER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

_ROI_HALF_SIZE = max(90, int(TRACKING_REACQUIRE_MAX_JUMP * 0.65))
_MIN_EXTENT_SMALL = 0.12
_MIN_SHORT_SIDE_SMALL = 2.5


# ---------------------------------------------------------------------------
# Main detector (keeps class name PaddleTracker for API compatibility)
# ---------------------------------------------------------------------------
class PaddleTracker:
    """Per-frame classical paddle detector with auto-calibration.

    Public interface is the same as the old multi-cue tracker so that
    ``main.py`` does not need to change its call sites.
    """

    def __init__(self):
        self._state: dict[bool, _PlayerDetectState] = {
            True: _PlayerDetectState(
                np.array([100, HEIGHT // 2], dtype=float),
                PADDLE_HSV_PRIORS["CYAN"], is_p1=True,
            ),
            False: _PlayerDetectState(
                np.array([WIDTH - 100, HEIGHT // 2], dtype=float),
                PADDLE_HSV_PRIORS["PINK"], is_p1=False,
            ),
        }

        # Frame buffers
        self._curr_bgr: np.ndarray | None = None
        self._curr_gray: np.ndarray | None = None
        self._edge_map: np.ndarray | None = None
        self._curr_lab: np.ndarray | None = None  # LAB color space image

        # Game mode flag
        self.in_game: bool = False

    def reset_to_priors(self) -> None:
        """Reset HSV ranges to broad priors and clear lock state.

        Called before auto-calibration so that stale saved profiles
        don't constrain the motion-guided colour learning.
        """
        for is_p1, key in [(True, "CYAN"), (False, "PINK")]:
            st = self._state[is_p1]
            prior = PADDLE_HSV_PRIORS[key]
            st.h_min = prior["h_min"]
            st.h_max = prior["h_max"]
            st.s_min = prior["s_min"]
            st.s_max = prior["s_max"]
            st.v_min = prior["v_min"]
            st.v_max = prior["v_max"]
            st.locked = False
            st.lock_counter = 0
            st.lab_ref = None

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------
    def begin_frame(self, frame_bgr: np.ndarray):
        """Call once per frame before any detection / calibration."""
        self._curr_bgr = frame_bgr
        self._curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Canny edge map (used for edge scoring)
        blurred_gray = cv2.GaussianBlur(self._curr_gray, (5, 5), 0)
        self._edge_map = cv2.Canny(blurred_gray, EDGE_CANNY_LOW, EDGE_CANNY_HIGH)

        # LAB color space (used for lighting-robust color scoring)
        self._curr_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)


        # Kalman predict step for both players (sets kalman_pos before detection)
        for is_p1 in (True, False):
            self._state[is_p1].kalman_predict()

    def end_frame(self):
        """Called after all detection calls.  Stores previous positions."""
        # prev_pos is updated inside track_paddle, nothing else needed
        pass

    # ------------------------------------------------------------------
    # Public properties (same interface as old tracker)
    # ------------------------------------------------------------------
    @property
    def p1_locked(self) -> bool:
        return self._state[True].locked

    @p1_locked.setter
    def p1_locked(self, value: bool):
        self._state[True].locked = value

    @property
    def p2_locked(self) -> bool:
        return self._state[False].locked

    @p2_locked.setter
    def p2_locked(self, value: bool):
        self._state[False].locked = value

    @property
    def both_locked(self) -> bool:
        return self.p1_locked and self.p2_locked

    def get_confidence(self, is_p1: bool) -> float:
        return self._state[is_p1].confidence

    # ------------------------------------------------------------------
    # HSV helpers
    # ------------------------------------------------------------------
    @staticmethod
    def blur_hsv(hsv: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(hsv, (HSV_BLUR_KSIZE, HSV_BLUR_KSIZE), 0)

    def _get_mask(self, hsv_blurred: np.ndarray, is_p1: bool) -> np.ndarray:
        st = self._state[is_p1]
        lower = np.array([st.h_min, st.s_min, st.v_min])
        upper = np.array([st.h_max, st.s_max, st.v_max])
        raw = cv2.inRange(hsv_blurred, lower, upper)

        # Main branch: robust cleanup for normal face-on blobs.
        mask_main = cv2.morphologyEx(raw, cv2.MORPH_OPEN, _KERN_OPEN)
        mask_main = cv2.morphologyEx(mask_main, cv2.MORPH_CLOSE, _KERN_CLOSE)

        # Thin branch: preserve edge-on blobs that can be erased by the
        # larger opening kernel above.
        mask_thin = cv2.morphologyEx(raw, cv2.MORPH_OPEN, _KERN_OPEN_THIN)
        mask_thin = cv2.morphologyEx(mask_thin, cv2.MORPH_CLOSE, _KERN_CLOSE_THIN)
        mask_thin = cv2.dilate(mask_thin, _KERN_OPEN_THIN, iterations=1)

        return cv2.bitwise_or(mask_main, mask_thin)

    @staticmethod
    def _contour_pixels(hsv: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """Sample mostly interior contour pixels to reduce boundary bleed."""
        cmask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(cmask, [contour], -1, 255, -1)

        inner = cv2.erode(cmask, _KERN_COLOR_INNER, iterations=1)
        use_mask = inner if np.count_nonzero(inner) >= 12 else cmask
        return hsv[use_mask > 0]

    @staticmethod
    def _purity_for_state(pixels: np.ndarray, st: _PlayerDetectState) -> float:
        if len(pixels) == 0:
            return 0.0
        lower = np.array([st.h_min, st.s_min, st.v_min])
        upper = np.array([st.h_max, st.s_max, st.v_max])
        in_range = np.all((pixels >= lower) & (pixels <= upper), axis=1)
        return float(np.mean(in_range))

    # ------------------------------------------------------------------
    # Per-contour feature scoring
    # ------------------------------------------------------------------
    def _color_score(self, contour: np.ndarray, hsv: np.ndarray, is_p1: bool) -> float:
        """HSV in-range purity + saturation strength + LAB distance."""
        st = self._state[is_p1]
        pixels = self._contour_pixels(hsv, contour)
        if len(pixels) < 5:
            return 0.0

        own_purity = self._purity_for_state(pixels, st)
        opp_purity = self._purity_for_state(pixels, self._state[not is_p1])
        purity = max(0.0, own_purity - 0.55 * opp_purity)

        avg_sat = float(np.mean(pixels[:, 1])) / 255.0
        hsv_score = purity * 0.65 + avg_sat * 0.25 + own_purity * 0.10

        # LAB distance scoring (lighting-robust)
        lab_score = self._lab_score(contour, is_p1)

        # Fuse HSV and LAB
        if lab_score >= 0:
            total_w = DETECT_COLOR_WEIGHT + DETECT_LAB_WEIGHT
            return (hsv_score * DETECT_COLOR_WEIGHT + lab_score * DETECT_LAB_WEIGHT) / total_w
        return hsv_score

    def _lab_score(self, contour: np.ndarray, is_p1: bool) -> float:
        """Score contour based on LAB color distance to reference."""
        st = self._state[is_p1]
        if st.lab_ref is None or self._curr_lab is None:
            return -1.0  # no reference yet

        # Sample LAB pixels inside contour
        cmask = np.zeros(self._curr_lab.shape[:2], dtype=np.uint8)
        cv2.drawContours(cmask, [contour], -1, 255, -1)
        lab_pixels = self._curr_lab[cmask > 0]
        if len(lab_pixels) < 5:
            return -1.0

        mean_lab = np.mean(lab_pixels, axis=0).astype(float)
        # Delta-E (simple Euclidean in LAB for speed)
        delta_e = np.sqrt(np.sum((mean_lab - st.lab_ref.astype(float)) ** 2))
        # Normalize: 0 at threshold, 1 at 0 distance
        return max(0.0, 1.0 - delta_e / LAB_DIST_THRESHOLD)

    def _edge_score(self, contour: np.ndarray) -> float:
        """Canny edge density along contour boundary."""
        if self._edge_map is None:
            return 0.5  # neutral if no edge map
        boundary = np.zeros(self._edge_map.shape, dtype=np.uint8)
        cv2.drawContours(boundary, [contour], -1, 255, 2)  # 2px boundary band
        n_boundary = np.count_nonzero(boundary)
        if n_boundary == 0:
            return 0.0
        n_edge = np.count_nonzero(cv2.bitwise_and(self._edge_map, boundary))
        return float(n_edge) / n_boundary

    @staticmethod
    def _shape_score(contour: np.ndarray) -> float:
        """Score from solidity, compactness, area, aspect ratio.

        Made deliberately lenient for edge-on views: small area and extreme
        aspect ratio are penalised gently so that colour + proximity can
        still carry the overall fused score.
        """
        area = cv2.contourArea(contour)
        if area < 1:
            return 0.0

        # Area appropriateness — lower floor (800) so small edge-on blobs
        # still get a decent score instead of being crushed
        area_norm = min(area / 800.0, 1.0)
        area_penalty = 1.0 if area <= CONTOUR_AREA_MAX else 0.2
        area_s = area_norm * area_penalty

        # Solidity & compactness — gentler denominators
        solidity, compactness = _contour_quality(contour)
        sol_s = min(solidity / 0.55, 1.0)
        comp_s = min(compactness / 0.35, 1.0)

        # Aspect ratio — gentler curve via square-root decay
        bx, by, bw, bh = cv2.boundingRect(contour)
        aspect = max(bw / bh, bh / bw) if (bh > 0 and bw > 0) else 99.0
        # sqrt-decay: aspect=1 -> 1.0, aspect=7 -> ~0.42, aspect=14 -> ~0.18
        asp_s = max(0.0, 1.0 - math.sqrt(max(aspect - 1.0, 0.0) / max(EDGE_ON_ASPECT_MAX - 1.0, 1.0)))

        return area_s * 0.25 + sol_s * 0.30 + comp_s * 0.20 + asp_s * 0.25

    @staticmethod
    def _proximity_score(contour: np.ndarray, prev_pos: np.ndarray) -> float:
        """Closeness to previous frame position (soft tie-breaker)."""
        bx, by, bw, bh = cv2.boundingRect(contour)
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        dist = math.sqrt((cx - prev_pos[0]) ** 2 + (cy - prev_pos[1]) ** 2)
        # Normalise: 0 at max jump, 1 at same position
        return max(0.0, 1.0 - dist / max(TRACKING_REACQUIRE_MAX_JUMP, 1.0))



    # ------------------------------------------------------------------
    # Main per-frame detection (replaces track_paddle)
    # ------------------------------------------------------------------
    def _detect(
        self, hsv: np.ndarray, hsv_blurred: np.ndarray, is_p1: bool,
    ) -> tuple[np.ndarray | None, np.ndarray, float]:
        """Detect paddle for one player.

        Returns ``(contour_or_None, center_xy, confidence)``.
        """
        st = self._state[is_p1]
        mask = self._get_mask(hsv_blurred, is_p1)

        # When the paddle has been missing for too long, skip ROI and
        # search the full frame to enable re-acquisition at any position.
        lost = st.miss_count > CONFIDENCE_GRACE_FRAMES

        candidates = []

        def _score_mask(mask_view: np.ndarray, x_off: int = 0, y_off: int = 0):
            contours, _ = cv2.findContours(
                mask_view, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )

            for c_local in contours:
                if x_off or y_off:
                    c = c_local.copy()
                    c[:, 0, 0] += x_off
                    c[:, 0, 1] += y_off
                else:
                    c = c_local

                area = cv2.contourArea(c)
                if area < MIN_CONTOUR_AREA_EDGEON or area > CONTOUR_AREA_MAX:
                    continue

                bx, by, bw, bh = cv2.boundingRect(c)
                if bw == 0 or bh == 0:
                    continue

                aspect = max(bw / bh, bh / bw)
                if aspect > EDGE_ON_ASPECT_MAX:
                    continue

                extent = area / float(max(bw * bh, 1))
                (_, _), (rw, rh), _ = cv2.minAreaRect(c)
                short_side = min(rw, rh)
                if area < MIN_CONTOUR_AREA and (extent < _MIN_EXTENT_SMALL or short_side < _MIN_SHORT_SIDE_SMALL):
                    continue

                # --- Feature scores ---
                c_score = self._color_score(c, hsv, is_p1)
                e_score = self._edge_score(c)
                s_score = self._shape_score(c)
                # When lost, don't penalise distance from stale prediction
                p_score = 1.0 if lost else self._proximity_score(c, st.prev_pos)

                fused = (
                    c_score * DETECT_COLOR_WEIGHT
                    + e_score * DETECT_EDGE_WEIGHT
                    + s_score * DETECT_SHAPE_WEIGHT
                    + p_score * DETECT_PROXIMITY_WEIGHT
                )

                # Edge-on boost: when contour is very thin (high aspect) but
                # colour match is solid, compensate for the low shape score
                # that thin views inevitably produce.
                if aspect > 4.0 and c_score >= 0.35:
                    edgeon_boost = min(0.10, c_score * 0.15)
                    fused += edgeon_boost

                cx = bx + bw / 2.0
                cy = by + bh / 2.0
                candidates.append((c, cx, cy, fused, c_score, e_score, s_score, p_score))

        # ROI-first search near Kalman-predicted position (or prev_pos).
        if not lost:
            kp = st.kalman_pos
            px = int(round(kp[0]))
            py = int(round(kp[1]))
            x0 = max(0, px - _ROI_HALF_SIZE)
            x1 = min(WIDTH, px + _ROI_HALF_SIZE)
            y0 = max(0, py - _ROI_HALF_SIZE)
            y1 = min(HEIGHT, py + _ROI_HALF_SIZE)

            if (x1 - x0) > 8 and (y1 - y0) > 8:
                _score_mask(mask[y0:y1, x0:x1], x0, y0)

            # Fallback: CSRT bounding box ROI if available and no candidates.
            if not candidates and st.csrt_bbox is not None and USE_CSRT_FALLBACK:
                bx, by, bw, bh = st.csrt_bbox
                pad = int(max(bw, bh) * 0.3)
                cx0 = max(0, bx - pad)
                cy0 = max(0, by - pad)
                cx1 = min(WIDTH, bx + bw + pad)
                cy1 = min(HEIGHT, by + bh + pad)
                if (cx1 - cx0) > 8 and (cy1 - cy0) > 8:
                    _score_mask(mask[cy0:cy1, cx0:cx1], cx0, cy0)

        # Fallback to full frame when ROI has no candidates or paddle is lost.
        if not candidates:
            _score_mask(mask)

        if not candidates:
            st.color_score = 0.0
            st.edge_score = 0.0
            st.shape_score = 0.0
            st.proximity_score = 0.0

            st.confidence = 0.0
            st.debug_reject_reason = "no candidates"
            return (None, st.smooth_pos.copy(), 0.0)

        # Pick best candidate
        best = max(candidates, key=lambda t: t[3])
        contour, cx, cy, fused, c_sc, e_sc, s_sc, p_sc = best

        # Store per-cue scores for diagnostics
        st.color_score = c_sc
        st.edge_score = e_sc
        st.shape_score = s_sc
        st.proximity_score = p_sc
        st.confidence = min(fused, 1.0)

        if fused < 0.05:
            st.debug_reject_reason = (
                f"low fused={fused:.2f} c={c_sc:.2f} e={e_sc:.2f} "
                f"s={s_sc:.2f} p={p_sc:.2f}"
            )
        else:
            st.debug_reject_reason = ""

        center = np.array([cx, cy], dtype=float)
        return (contour, center, st.confidence)

    def _refine_contour_watershed(self, frame_bgr: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """Refines the rough HSV contour using cv2.watershed for exact edges."""
        if frame_bgr is None or contour is None:
            return contour

        x, y, w, h = cv2.boundingRect(contour)
        PAD = 20
        H, W = frame_bgr.shape[:2]
        x1, y1 = max(0, x - PAD), max(0, y - PAD)
        x2, y2 = min(W, x + w + PAD), min(H, y + h + PAD)
        
        if x2 - x1 < 5 or y2 - y1 < 5:
            return contour

        roi_bgr = frame_bgr[y1:y2, x1:x2].copy()
        
        local_contour = contour.copy()
        local_contour[:, 0, 0] -= x1
        local_contour[:, 0, 1] -= y1
        
        mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [local_contour], -1, 255, -1)
        
        kernel = np.ones((5,5), np.uint8)
        sure_fg = cv2.erode(mask, kernel, iterations=2)
        sure_bg_mask = cv2.dilate(mask, kernel, iterations=3)
        sure_bg = cv2.bitwise_not(sure_bg_mask)
        
        markers = np.zeros(roi_bgr.shape[:2], dtype=np.int32)
        markers[sure_bg == 255] = 1
        markers[sure_fg == 255] = 2
        
        try:
            cv2.watershed(roi_bgr, markers)
        except Exception:
            return contour
        
        refined_mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
        refined_mask[markers == 2] = 255
        
        refined_contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not refined_contours:
            return contour
            
        best_refined = max(refined_contours, key=cv2.contourArea)
        best_refined[:, 0, 0] += x1
        best_refined[:, 0, 1] += y1
        
        return best_refined

    # ------------------------------------------------------------------
    # Public tracking entry point (same signature as old tracker)
    # ------------------------------------------------------------------
    def track_paddle(
        self, hsv_blurred: np.ndarray, is_p1: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Detect paddle and return ``(smoothed_position, contour_or_None)``."""
        st = self._state[is_p1]

        # Build raw HSV from blurred (needed for color scoring)
        if self._curr_bgr is not None:
            hsv_raw = cv2.cvtColor(self._curr_bgr, cv2.COLOR_BGR2HSV)
        else:
            hsv_raw = hsv_blurred

        # --- CSRT update (runs every frame for visual continuity) ---
        if USE_CSRT_FALLBACK and st.csrt is not None and self._curr_bgr is not None:
            ok, bbox = st.csrt.update(self._curr_bgr)
            if ok:
                st.csrt_bbox = tuple(int(v) for v in bbox)
            else:
                st.csrt_bbox = None
        else:
            st.csrt_bbox = None

        # --- Per-frame multi-cue detection ---
        contour, center, conf = self._detect(hsv_raw, hsv_blurred, is_p1)

        if contour is not None and self._curr_bgr is not None:
            contour = self._refine_contour_watershed(self._curr_bgr, contour)

        # --- Kalman correction / miss tracking ---
        if conf >= CONFIDENCE_HOLD and contour is not None:
            bx, by, bw, bh = cv2.boundingRect(contour)
            st.kalman_correct(center[0], center[1], float(bw), float(bh))

            # EMA smoothing
            alpha = EMA_ALPHA
            st.smooth_pos = alpha * center + (1.0 - alpha) * st.smooth_pos
            st.prev_pos = center.copy()
            st.miss_count = 0

            # Update last_area with EMA for size continuity guard
            det_area = float(cv2.contourArea(contour))
            if st.last_area > 0:
                st.last_area = 0.7 * st.last_area + 0.3 * det_area
            else:
                st.last_area = det_area
            st.last_seen_time = _time.time()

            # --- CSRT re-initialization on good detections ---
            if USE_CSRT_FALLBACK and self._curr_bgr is not None:
                st.csrt_good_count += 1
                if st.csrt_good_count >= CSRT_REINIT_INTERVAL or st.csrt is None:
                    st.csrt_good_count = 0
                    pad = 15
                    rx = max(0, bx - pad)
                    ry = max(0, by - pad)
                    rw = min(bw + 2 * pad, WIDTH - rx)
                    rh = min(bh + 2 * pad, HEIGHT - ry)
                    if rw > 5 and rh > 5:
                        try:
                            st.csrt = cv2.TrackerCSRT_create()
                            st.csrt.init(self._curr_bgr, (rx, ry, rw, rh))
                        except Exception:
                            st.csrt = None

            # --- Update LAB reference from good detections ---
            if self._curr_lab is not None and st.lab_ref is None:
                cmask = np.zeros(self._curr_lab.shape[:2], dtype=np.uint8)
                cv2.drawContours(cmask, [contour], -1, 255, -1)
                lab_px = self._curr_lab[cmask > 0]
                if len(lab_px) >= 10:
                    st.lab_ref = np.mean(lab_px, axis=0)
        else:
            st.miss_count += 1
            st.kalman_mark_miss()
            # Use Kalman prediction for smooth_pos during misses
            st.smooth_pos = st.kalman_pos.copy()

        # Clamp to frame
        sx = float(max(0.0, min(st.smooth_pos[0], float(WIDTH))))
        sy = float(max(0.0, min(st.smooth_pos[1], float(HEIGHT))))
        out_pos = np.array([sx, sy], dtype=float)

        # --- Temporal contour smoothing ---
        if contour is not None:
            st.contour_history.append(contour)
        smoothed_contour = self._smooth_contour(st) if contour is not None else None

        # Use smoothed contour for output
        use_contour = smoothed_contour if smoothed_contour is not None else contour

        # PARITY: Hysteresis confidence gate with grace frames.
        # During gameplay, NEVER drop the contour — always hold the
        # last known contour to avoid visual gaps.
        if st._contour_held:
            if conf >= CONFIDENCE_HOLD and use_contour is not None:
                st._grace_counter = CONFIDENCE_GRACE_FRAMES
                st._last_contour = use_contour
                out_contour = use_contour
            elif st._grace_counter > 0:
                st._grace_counter -= 1
                out_contour = st._last_contour
            elif self.in_game:
                # In-game: keep showing last contour (never drop)
                out_contour = st._last_contour
            else:
                st._contour_held = False
                st._last_contour = None
                out_contour = None
        else:
            if conf >= TRACKING_CONFIDENCE_MIN and use_contour is not None:
                st._contour_held = True
                st._grace_counter = CONFIDENCE_GRACE_FRAMES
                st._last_contour = use_contour
                out_contour = use_contour
            else:
                out_contour = None

        return out_pos, out_contour

    @staticmethod
    def _resample_contour(contour: np.ndarray, n_points: int) -> np.ndarray:
        """Resample contour to a fixed number of evenly-spaced points."""
        contour_2d = contour.reshape(-1, 2).astype(float)
        if len(contour_2d) < 3:
            return contour
        # Arc-length parameterization
        diffs = np.diff(contour_2d, axis=0)
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        cum = np.concatenate([[0], np.cumsum(dists)])
        total = cum[-1]
        if total < 1e-6:
            return contour
        # Interpolate
        targets = np.linspace(0, total, n_points, endpoint=False)
        xs = np.interp(targets, cum, contour_2d[:, 0])
        ys = np.interp(targets, cum, contour_2d[:, 1])
        resampled = np.stack([xs, ys], axis=1).astype(np.int32)
        return resampled.reshape(-1, 1, 2)

    @staticmethod
    def _smooth_contour(st: '_PlayerDetectState') -> np.ndarray | None:
        """Blend last N contour shapes for temporal smoothing."""
        history = st.contour_history
        if len(history) == 0:
            return None
        if len(history) == 1:
            return history[0]

        n = CONTOUR_RESAMPLE_POINTS
        resampled = []
        for c in history:
            r = PaddleTracker._resample_contour(c, n)
            if r.shape[0] == n:
                resampled.append(r.reshape(n, 2).astype(float))

        if not resampled:
            return history[-1]

        # Weighted average (newest = highest weight)
        weights = np.array([0.5 ** i for i in range(len(resampled) - 1, -1, -1)])
        weights /= weights.sum()

        blended = np.zeros((n, 2), dtype=float)
        for w, pts in zip(weights, resampled):
            blended += w * pts

        return blended.astype(np.int32).reshape(-1, 1, 2)

    # ------------------------------------------------------------------
    # Calibration (runs during STATE_AUTODETECT)
    # ------------------------------------------------------------------
    def calibrate_step(self, hsv: np.ndarray, hsv_blurred: np.ndarray):
        for is_p1 in (True, False):
            st = self._state[is_p1]
            if st.locked:
                continue

            mask = self._get_mask(hsv_blurred, is_p1)
            # No motion mask during calibration — broad HSV priors are
            # selective enough, and MOG2 needs warm-up time that delays
            # detection.
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )

            best_contour = None
            best_score = 0.0
            best_is_strong = False
            n_rejected_area = 0
            n_rejected_shape = 0
            n_rejected_sat = 0

            sat_floor = PINK_CALIB_SAT_MIN if not is_p1 else 80

            for c in contours:
                area = cv2.contourArea(c)
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw == 0 or bh == 0:
                    continue
                aspect = max(bw / bh, bh / bw)

                is_strong = _passes_quality_strict(c)
                if not is_strong:
                    if area < MIN_CONTOUR_AREA_EDGEON or area > CONTOUR_AREA_MAX:
                        n_rejected_area += 1
                        continue
                    if aspect > EDGE_ON_ASPECT_MAX:
                        n_rejected_shape += 1
                        continue

                # Colour purity
                pixels = self._contour_pixels(hsv, c)
                if len(pixels) < 10:
                    continue
                avg_sat = float(np.mean(pixels[:, 1]))
                if avg_sat < sat_floor:
                    n_rejected_sat += 1
                    continue
                own_purity = self._purity_for_state(pixels, st)
                opp_purity = self._purity_for_state(pixels, self._state[not is_p1])
                color_sep = max(0.0, own_purity - 0.5 * opp_purity)
                if color_sep < 0.10:
                    continue

                # Edge evidence bonus (helps separate real paddle from
                # ambient colour match)
                e_bonus = 1.0
                if self._edge_map is not None:
                    e_sc = self._edge_score(c)
                    e_bonus = 0.5 + e_sc  # range [0.5, 1.5]

                solidity, _ = _contour_quality(c)
                quality_bonus = 1.5 if is_strong else 0.8
                score = area * solidity * (avg_sat / 255.0) * quality_bonus * e_bonus * (0.3 + color_sep)
                if score > best_score:
                    best_score = score
                    best_contour = c
                    best_is_strong = is_strong

            if best_contour is not None:
                if best_is_strong:
                    self._refine_hsv_from_contour(hsv, best_contour, is_p1)
                st.enforce_floors()

                bx, by, bw, bh = cv2.boundingRect(best_contour)
                cx_f = bx + bw // 2
                cy_f = by + bh // 2
                st.prev_pos = np.array([cx_f, cy_f], dtype=float)
                st.smooth_pos = st.prev_pos.copy()

                increment = 1 if best_is_strong else max(1, CALIB_LOCK_FRAMES // 50)
                st.lock_counter = min(st.lock_counter + increment, CALIB_LOCK_FRAMES)
                st.debug_lock_fail_reason = ""

                if st.lock_counter >= CALIB_LOCK_FRAMES:
                    st.locked = True
            else:
                st.lock_counter = max(0, st.lock_counter - CALIB_LOCK_DECAY)
                parts = []
                if not contours:
                    parts.append("no HSV blobs")
                else:
                    if n_rejected_area:
                        parts.append(f"area reject x{n_rejected_area}")
                    if n_rejected_shape:
                        parts.append(f"shape reject x{n_rejected_shape}")
                    if n_rejected_sat:
                        parts.append(f"low sat x{n_rejected_sat}")
                    if not parts:
                        parts.append("quality too low")
                st.debug_lock_fail_reason = "; ".join(parts)

    def _refine_hsv_from_contour(
        self, hsv: np.ndarray, contour: np.ndarray, is_p1: bool,
    ):
        pixels = self._contour_pixels(hsv, contour)
        if len(pixels) < 20:
            return
        st = self._state[is_p1]
        hues = pixels[:, 0].astype(float)
        sats = pixels[:, 1].astype(float)
        vals = pixels[:, 2].astype(float)

        sat_floor = PINK_CALIB_SAT_MIN if not is_p1 else 80
        if float(np.median(sats)) < sat_floor:
            return

        # Saturation-weighted hue centre
        weights = sats / (sats.sum() + 1e-6)
        h_weighted = float(np.sum(hues * weights))
        new_h_min = max(0, int(h_weighted - HSV_REFINE_HUE_TOL))
        new_h_max = min(179, int(h_weighted + HSV_REFINE_HUE_TOL))

        high_sat_mask = sats >= np.percentile(sats, 30)
        if np.sum(high_sat_mask) >= 10:
            sats_hs = sats[high_sat_mask]
            vals_hs = vals[high_sat_mask]
        else:
            sats_hs = sats
            vals_hs = vals

        new_s_min = max(0, int(np.percentile(sats_hs, 10) - HSV_REFINE_SV_TOL))
        new_s_max = min(255, int(np.percentile(sats_hs, 90) + HSV_REFINE_SV_TOL))
        new_v_min = max(0, int(np.percentile(vals_hs, 10) - HSV_REFINE_SV_TOL))
        new_v_max = min(255, int(np.percentile(vals_hs, 90) + HSV_REFINE_SV_TOL))

        b = 0.35
        st.h_min = int(st.h_min * (1 - b) + new_h_min * b)
        st.h_max = int(st.h_max * (1 - b) + new_h_max * b)
        st.s_min = int(st.s_min * (1 - b) + new_s_min * b)
        st.s_max = int(st.s_max * (1 - b) + new_s_max * b)
        st.v_min = int(st.v_min * (1 - b) + new_v_min * b)
        st.v_max = int(st.v_max * (1 - b) + new_v_max * b)
        st.enforce_floors()

    def calibration_progress(self, is_p1: bool) -> float:
        st = self._state[is_p1]
        return 1.0 if st.locked else min(st.lock_counter / CALIB_LOCK_FRAMES, 1.0)



    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self):
        pass
