"""UI layer: home menu, pause menu, results screen, HUD, hover/hold buttons."""

import cv2
import numpy as np
import time

from .config import (
    WIDTH,
    HEIGHT,
    HOVER_DURATION,
    PAUSE_HOLD_DURATION,
    PAUSE_TOP_THRESHOLD,
    COLOR_P1,
    COLOR_P2,
    COLOR_WHITE,
    COLOR_BLACK,
    COLOR_GRAY,
    DIFFICULTY_NAMES,
    COMBO_MIN_HITS,
    COMBO_BANNER_DURATION,
    ENDING_WARNING_SECONDS,
)


# ---------------------------------------------------------------------------
# Hover button
# ---------------------------------------------------------------------------
class HoverButton:
    """A UI button activated by holding a paddle over it for HOVER_DURATION."""

    def __init__(self, x: int, y: int, w: int, h: int, label: str, color=(0, 255, 0)):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label
        self.color = color
        self.hover_start: float | None = None

    def _is_hovered(self, positions) -> bool:
        for pos in positions:
            if self.x < pos[0] < self.x + self.w and self.y < pos[1] < self.y + self.h:
                return True
        return False

    def update(self, positions) -> float:
        if self._is_hovered(positions):
            if self.hover_start is None:
                self.hover_start = time.time()
            return min((time.time() - self.hover_start) / HOVER_DURATION, 1.0)
        self.hover_start = None
        return 0.0

    def draw(self, frame: np.ndarray, progress: float = 0.0):
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.x, self.y), (self.x + self.w, self.y + self.h), self.color, -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), self.color, 2)
        ts = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        tx = self.x + (self.w - ts[0]) // 2
        ty = self.y + (self.h + ts[1]) // 2
        cv2.putText(frame, self.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        if progress > 0:
            cx = self.x + self.w // 2
            cy = self.y + self.h // 2
            rx = self.w // 2 + 10
            ry = self.h // 2 + 10
            cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, int(360 * progress), COLOR_WHITE, 3)


# ---------------------------------------------------------------------------
# UI manager
# ---------------------------------------------------------------------------
class UIManager:
    """Owns every menu screen, the HUD, and the pause-gesture detector."""

    def __init__(self):
        btn_w, btn_h = 180, 50
        cx = WIDTH // 2 - btn_w // 2

        # --- Home screen --------------------------------------------------
        self.home_buttons: dict[str, HoverButton] = {
            "start": HoverButton(cx, 160, btn_w, btn_h, "START GAME", (0, 200, 0)),
        }
        diff_y = 280
        diff_w = 120
        diff_colors = [(0, 200, 0), (0, 200, 200), (0, 100, 255), (0, 0, 255)]
        self.diff_buttons: dict[str, HoverButton] = {}
        for i, name in enumerate(DIFFICULTY_NAMES):
            x = 40 + i * (diff_w + 20)
            self.diff_buttons[f"diff_{i}"] = HoverButton(x, diff_y, diff_w, 40, name, diff_colors[i])

        self.home_buttons["toggle_diff"] = HoverButton(
            WIDTH // 2 - 90, 230, 180, 35, "CHANGE DIFFICULTY", (180, 140, 0),
        )

        self.selected_difficulty: int = 0
        self._difficulty_visible: bool = True

        # --- Pause screen --------------------------------------------------
        px = WIDTH // 2 - 90
        self.pause_buttons: dict[str, HoverButton] = {
            "resume":  HoverButton(px, 160, 180, 50, "RESUME",  (0, 200, 0)),
            "restart": HoverButton(px, 230, 180, 50, "RESTART", (0, 200, 200)),
            "home":    HoverButton(px, 300, 180, 50, "HOME",    (0, 100, 255)),
        }

        # --- Results screen ------------------------------------------------
        self.result_buttons: dict[str, HoverButton] = {
            "replay":     HoverButton(cx, 300, btn_w, btn_h, "WATCH REPLAY", (200, 100, 0)),
            "play_again": HoverButton(cx, 370, btn_w, btn_h, "PLAY AGAIN",   (0, 200, 0)),
            "home":       HoverButton(cx, 440, btn_w, btn_h, "HOME",         (0, 100, 255)),
        }

        # --- Dynamic player colors -----------------------------------------
        self.p1_color: tuple = COLOR_P1
        self.p2_color: tuple = COLOR_P2

        # --- Pause gesture -------------------------------------------------
        self.pause_hold_start: float | None = None

    # ------------------------------------------------------------------
    # Calibration / auto-detect screen (new: no selection box)
    # ------------------------------------------------------------------
    def draw_calibration(
        self,
        frame: np.ndarray,
        p1_progress: float,
        p2_progress: float,
        p1_locked: bool,
        p2_locked: bool,
    ) -> None:
        """Draw the automatic calibration overlay.

        Shows a simple status screen while HSV auto-calibration runs.
        No user interaction is required — it auto-transitions once
        both paddles are locked.
        """
        # Semi-transparent dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # Title
        self._outlined_text(frame, "AUTO CALIBRATING...", (WIDTH // 2 - 170, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)
        self._outlined_text(frame, "Hold your paddles visible to the camera",
                            (WIDTH // 2 - 210, 90),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, COLOR_GRAY, 1)

        # P1 status (cyan)
        p1_bar_x = 80
        p1_bar_y = 160
        bar_w = 200
        bar_h = 30

        p1_label = "P1 (CYAN)"
        p1_status = "LOCKED" if p1_locked else f"Detecting... {int(p1_progress * 100)}%"
        p1_status_col = (0, 255, 0) if p1_locked else COLOR_P1

        cv2.putText(frame, p1_label, (p1_bar_x, p1_bar_y - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_P1, 2)
        cv2.rectangle(frame, (p1_bar_x, p1_bar_y),
                       (p1_bar_x + bar_w, p1_bar_y + bar_h), COLOR_GRAY, -1)
        fill_w = int(bar_w * p1_progress)
        cv2.rectangle(frame, (p1_bar_x, p1_bar_y),
                       (p1_bar_x + fill_w, p1_bar_y + bar_h), p1_status_col, -1)
        cv2.rectangle(frame, (p1_bar_x, p1_bar_y),
                       (p1_bar_x + bar_w, p1_bar_y + bar_h), COLOR_WHITE, 1)
        cv2.putText(frame, p1_status, (p1_bar_x + bar_w + 15, p1_bar_y + 22),
                     cv2.FONT_HERSHEY_PLAIN, 1.2, p1_status_col, 1)

        # Animated dot for P1 (pulsing) if not locked
        if not p1_locked:
            pulse = int(abs(np.sin(time.time() * 4)) * 6)
            cv2.circle(frame, (p1_bar_x - 20, p1_bar_y + bar_h // 2),
                       6 + pulse, COLOR_P1, -1)

        # P2 status (pink)
        p2_bar_x = 80
        p2_bar_y = 240

        p2_label = "P2 (PINK)"
        p2_status = "LOCKED" if p2_locked else f"Detecting... {int(p2_progress * 100)}%"
        p2_status_col = (0, 255, 0) if p2_locked else COLOR_P2

        cv2.putText(frame, p2_label, (p2_bar_x, p2_bar_y - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_P2, 2)
        cv2.rectangle(frame, (p2_bar_x, p2_bar_y),
                       (p2_bar_x + bar_w, p2_bar_y + bar_h), COLOR_GRAY, -1)
        fill_w = int(bar_w * p2_progress)
        cv2.rectangle(frame, (p2_bar_x, p2_bar_y),
                       (p2_bar_x + fill_w, p2_bar_y + bar_h), p2_status_col, -1)
        cv2.rectangle(frame, (p2_bar_x, p2_bar_y),
                       (p2_bar_x + bar_w, p2_bar_y + bar_h), COLOR_WHITE, 1)
        cv2.putText(frame, p2_status, (p2_bar_x + bar_w + 15, p2_bar_y + 22),
                     cv2.FONT_HERSHEY_PLAIN, 1.2, p2_status_col, 1)

        if not p2_locked:
            pulse = int(abs(np.sin(time.time() * 4 + 1.5)) * 6)
            cv2.circle(frame, (p2_bar_x - 20, p2_bar_y + bar_h // 2),
                       6 + pulse, COLOR_P2, -1)

        # Hint
        self._outlined_text(frame, "Show your CYAN and PINK paddles to the camera",
                            (WIDTH // 2 - 230, HEIGHT - 60),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, COLOR_GRAY, 1)

        # Check mark for locked players
        if p1_locked:
            cv2.putText(frame, "OK", (p1_bar_x - 30, p1_bar_y + 22),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if p2_locked:
            cv2.putText(frame, "OK", (p2_bar_x - 30, p2_bar_y + 22),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ------------------------------------------------------------------
    # Home screen
    # ------------------------------------------------------------------
    @staticmethod
    def _outlined_text(frame, text, pos, font, scale, color, thickness,
                       outline_color=COLOR_BLACK, outline_thickness=None):
        if outline_thickness is None:
            outline_thickness = thickness + 3
        cv2.putText(frame, text, pos, font, scale, outline_color, outline_thickness)
        cv2.putText(frame, text, pos, font, scale, color, thickness)

    def draw_home(self, frame: np.ndarray, p1_pos, p2_pos) -> str | None:
        positions = [p1_pos, p2_pos]

        self._outlined_text(frame, "COLOR WAR PONG", (WIDTH // 2 - 180, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_WHITE, 3)
        self._outlined_text(frame, "Hover paddle to select", (WIDTH // 2 - 130, 130),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, COLOR_GRAY, 1)

        self._outlined_text(
            frame,
            f"Difficulty: {DIFFICULTY_NAMES[self.selected_difficulty]}",
            (WIDTH // 2 - 100, 225),
            cv2.FONT_HERSHEY_PLAIN, 1.2, COLOR_WHITE, 1,
        )

        action = None

        for key, btn in self.home_buttons.items():
            if key == "toggle_diff" and self._difficulty_visible:
                continue
            prog = btn.update(positions)
            btn.draw(frame, prog)
            if prog >= 1.0:
                btn.hover_start = None
                if key == "start":
                    action = "start"
                elif key == "toggle_diff":
                    self._difficulty_visible = True

        if self._difficulty_visible:
            for key, btn in self.diff_buttons.items():
                prog = btn.update(positions)
                btn.draw(frame, prog)
                if prog >= 1.0:
                    btn.hover_start = None
                    self.selected_difficulty = int(key.split("_")[1])
                    self._difficulty_visible = False

        # Paddle cursors
        cv2.circle(frame, (int(p1_pos[0]), int(p1_pos[1])), 12, self.p1_color, -1)
        cv2.circle(frame, (int(p2_pos[0]), int(p2_pos[1])), 12, self.p2_color, -1)

        self._outlined_text(
            frame,
            "Hold both paddles at TOP to PAUSE during game",
            (WIDTH // 2 - 220, HEIGHT - 30),
            cv2.FONT_HERSHEY_PLAIN, 1, COLOR_GRAY, 1,
        )
        return action

    # ------------------------------------------------------------------
    # Pause screen
    # ------------------------------------------------------------------
    def draw_pause(self, frame: np.ndarray, p1_pos, p2_pos) -> str | None:
        overlay = np.zeros_like(frame)
        cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)
        cv2.putText(frame, "PAUSED", (WIDTH // 2 - 80, 100),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_WHITE, 3)

        positions = [p1_pos, p2_pos]
        action = None
        for key, btn in self.pause_buttons.items():
            prog = btn.update(positions)
            btn.draw(frame, prog)
            if prog >= 1.0:
                btn.hover_start = None
                action = key

        cv2.circle(frame, (int(p1_pos[0]), int(p1_pos[1])), 12, self.p1_color, -1)
        cv2.circle(frame, (int(p2_pos[0]), int(p2_pos[1])), 12, self.p2_color, -1)
        return action

    # ------------------------------------------------------------------
    # Results screen
    # ------------------------------------------------------------------
    def draw_results(
        self,
        frame: np.ndarray,
        paint_canvas: np.ndarray,
        winner_text: str,
        p1_pct: int,
        p2_pct: int,
        p1_pos,
        p2_pos,
    ) -> tuple[str | None, np.ndarray]:
        res = cv2.addWeighted(frame, 0.3, paint_canvas, 0.7, 0)

        cv2.rectangle(res, (80, 60), (WIDTH - 80, HEIGHT - 20), COLOR_BLACK, -1)
        cv2.rectangle(res, (80, 60), (WIDTH - 80, HEIGHT - 20), COLOR_WHITE, 2)

        cv2.putText(res, winner_text, (WIDTH // 2 - 140, 130),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_WHITE, 3)

        bar_w = 300
        cv2.putText(res, f"P1: {p1_pct}%", (120, 180), cv2.FONT_HERSHEY_PLAIN, 1.5, self.p1_color, 2)
        cv2.rectangle(res, (120, 190), (120 + int(p1_pct / 100 * bar_w), 215), self.p1_color, -1)

        cv2.putText(res, f"P2: {p2_pct}%", (120, 250), cv2.FONT_HERSHEY_PLAIN, 1.5, self.p2_color, 2)
        cv2.rectangle(res, (120, 260), (120 + int(p2_pct / 100 * bar_w), 285), self.p2_color, -1)

        positions = [p1_pos, p2_pos]
        action = None
        for key, btn in self.result_buttons.items():
            prog = btn.update(positions)
            btn.draw(res, prog)
            if prog >= 1.0:
                btn.hover_start = None
                action = key

        cv2.circle(res, (int(p1_pos[0]), int(p1_pos[1])), 12, self.p1_color, -1)
        cv2.circle(res, (int(p2_pos[0]), int(p2_pos[1])), 12, self.p2_color, -1)
        return action, res

    # ------------------------------------------------------------------
    # Pause target rectangle
    # ------------------------------------------------------------------
    def _draw_pause_target(self, frame: np.ndarray):
        margin = 40
        x1, y1 = margin, 2
        x2, y2 = WIDTH - margin, PAUSE_TOP_THRESHOLD
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 60, 60), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        dash_len = 10
        for x in range(x1, x2, dash_len * 2):
            cv2.line(frame, (x, y1), (min(x + dash_len, x2), y1), COLOR_GRAY, 1)
            cv2.line(frame, (x, y2), (min(x + dash_len, x2), y2), COLOR_GRAY, 1)
        for y in range(y1, y2, dash_len * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash_len, y2)), COLOR_GRAY, 1)
            cv2.line(frame, (x2, y), (x2, min(y + dash_len, y2)), COLOR_GRAY, 1)
        cv2.putText(frame, "PAUSE ZONE", (x2 - 100, y2 - 5),
                     cv2.FONT_HERSHEY_PLAIN, 0.8, COLOR_GRAY, 1)

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------
    def draw_hud(self, frame: np.ndarray, time_left: int, difficulty: int,
                 level: int = 1, level_up_until: float = 0,
                 combo_count: int = 0, combo_owner: str | None = None,
                 combo_banner_until: float = 0,
                 p1_color: tuple = COLOR_P1, p2_color: tuple = COLOR_P2):
        self._draw_pause_target(frame)

        now = time.time()
        if time_left <= ENDING_WARNING_SECONDS and time_left > 0:
            pulse = abs(np.sin(now * 6))
            timer_bg = (0, 0, int(180 * pulse))
            timer_border = (0, 0, 255)
            timer_text_col = (0, 0, 255)
        else:
            timer_bg = COLOR_BLACK
            timer_border = COLOR_WHITE
            timer_text_col = COLOR_WHITE

        cv2.rectangle(frame, (WIDTH // 2 - 40, 5), (WIDTH // 2 + 40, 45), timer_bg, -1)
        cv2.rectangle(frame, (WIDTH // 2 - 40, 5), (WIDTH // 2 + 40, 45), timer_border, 1)
        cv2.putText(frame, str(time_left), (WIDTH // 2 - 18, 38),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, timer_text_col, 2)
        cv2.putText(frame, DIFFICULTY_NAMES[difficulty], (10, 25),
                     cv2.FONT_HERSHEY_PLAIN, 1.2, COLOR_GRAY, 1)
        cv2.putText(frame, f"LVL {level}", (WIDTH - 80, 25),
                     cv2.FONT_HERSHEY_PLAIN, 1.2, COLOR_WHITE, 1)

        # Combo banner
        if combo_count >= COMBO_MIN_HITS and now < combo_banner_until:
            remaining = combo_banner_until - now
            fade = min(1.0, remaining / (COMBO_BANNER_DURATION * 0.5))
            banner_color = p1_color if combo_owner == "P1" else p2_color
            scale = 0.9 + 0.15 * abs(np.sin(now * 8))
            text = f"COMBO x{combo_count}!"
            ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)[0]
            tx = WIDTH // 2 - ts[0] // 2
            ty = HEIGHT - 50
            outline_col = tuple(int(c * fade) for c in COLOR_BLACK)
            text_col = tuple(int(c * fade) for c in banner_color)
            cv2.putText(frame, text, (tx, ty),
                         cv2.FONT_HERSHEY_SIMPLEX, scale, outline_col, 5)
            cv2.putText(frame, text, (tx, ty),
                         cv2.FONT_HERSHEY_SIMPLEX, scale, text_col, 3)

        # Level-up overlay
        if now < level_up_until:
            remaining = level_up_until - now
            from .config import LEVEL_OVERLAY_DURATION
            fade = min(1.0, remaining / (LEVEL_OVERLAY_DURATION * 0.6))
            overlay = frame.copy()
            cv2.rectangle(overlay, (WIDTH // 2 - 120, HEIGHT // 2 - 35),
                           (WIDTH // 2 + 120, HEIGHT // 2 + 35), COLOR_BLACK, -1)
            cv2.addWeighted(overlay, 0.7 * fade, frame, 1.0 - 0.7 * fade, 0, frame)
            alpha_col = tuple(int(c * fade) for c in (0, 255, 255))
            cv2.putText(frame, f"LEVEL {level}!", (WIDTH // 2 - 80, HEIGHT // 2 + 12),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.2, alpha_col, 3)

    # ------------------------------------------------------------------
    # Pause gesture
    # ------------------------------------------------------------------
    def check_pause_gesture(self, p1_pos, p2_pos) -> bool:
        if p1_pos[1] < PAUSE_TOP_THRESHOLD and p2_pos[1] < PAUSE_TOP_THRESHOLD:
            if self.pause_hold_start is None:
                self.pause_hold_start = time.time()
            elif time.time() - self.pause_hold_start >= PAUSE_HOLD_DURATION:
                self.pause_hold_start = None
                return True
        else:
            self.pause_hold_start = None
        return False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def reset_hover_states(self):
        for btn in self.home_buttons.values():
            btn.hover_start = None
        for btn in self.diff_buttons.values():
            btn.hover_start = None
        for btn in self.pause_buttons.values():
            btn.hover_start = None
        for btn in self.result_buttons.values():
            btn.hover_start = None
        self.pause_hold_start = None
