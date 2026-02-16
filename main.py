"""Main application: camera loop, state machine, frame composition."""

import cv2
import numpy as np
import time
import threading
import random

from .config import (
    WIDTH,
    HEIGHT,
    PADDLE_RADIUS,
    PAUSE_HOLD_DURATION,
    GAME_DURATION,
    COLOR_P1,
    COLOR_P2,
    COLOR_POWERUP,
    COLOR_BLACK,
    COLOR_GRAY,
    COLOR_MIDLINE,
    COLOR_WHITE,
    STATE_AUTODETECT,
    STATE_HOME,
    STATE_GAME,
    STATE_PAUSE,
    STATE_RESULTS,
    STATE_REPLAY,
    REPLAY_SAMPLE_EVERY,
    REPLAY_DURATION,
    PROGRESS_BEEP_INTERVAL,
    PROGRESS_BEEP_MIN_FREQ,
    PROGRESS_BEEP_MAX_FREQ,
    ENDING_WARNING_SECONDS,
    ENDING_JINGLE_INTERVAL,
    SHAKE_DURATION,
    SHAKE_INTENSITY_HIT,
    SHAKE_INTENSITY_LEVEL,
    TRACKING_CONFIDENCE_MIN,
)
from .tracking import PaddleTracker
from .game import GameEngine
from .ui import UIManager
from .challenges import ChallengeManager


class ColorWarApp:
    """Top-level controller that wires tracking, engine, UI, and challenges."""

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, WIDTH)
        self.cap.set(4, HEIGHT)

        self.tracker = PaddleTracker()
        self._load_calibration_profile()
        self.engine = GameEngine()
        self.ui = UIManager()
        self.challenges = ChallengeManager()

        # Initialise pygame mixer for sound effects
        self._sound_ready = False
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self._sound_ready = True
        except Exception:
            pass

        self.state: str = STATE_AUTODETECT

        # Player positions and contours
        self.p1_pos = np.array([100, HEIGHT // 2], dtype=float)
        self.p2_pos = np.array([WIDTH - 100, HEIGHT // 2], dtype=float)
        self.p1_prev = self.p1_pos.copy()
        self.p2_prev = self.p2_pos.copy()
        self.p1_contour: np.ndarray | None = None
        self.p2_contour: np.ndarray | None = None

        # Pause bookkeeping
        self._pause_start: float = 0.0

        # Brief help overlay timer
        self._help_show_until: float = 0.0

        # Pre-start warning state
        self._p1_was_wrong_side: bool = False
        self._p2_was_wrong_side: bool = False

        # Audio scheduler state
        self._next_progress_beep: float = 0.0
        self._next_ending_jingle: float = 0.0
        self._ending_jingle_active: bool = False

        # Screen shake state
        self._shake_until: float = 0.0
        self._shake_intensity: int = 0

        # Timelapse replay recording
        self._replay_frames: list[np.ndarray] = []
        self._game_frame_idx: int = 0

        # Replay playback state
        self._replay_playback_idx: float = 0.0
        self._replay_step: float = 0.0

    # ------------------------------------------------------------------
    # Calibration profile loader (optional — from calibration_app.py)
    # ------------------------------------------------------------------
    def _load_calibration_profile(self):
        """Load tuned HSV ranges from a calibration profile if one exists."""
        try:
            from .calibration_app import load_profile, _PROFILE_PATH
            if _PROFILE_PATH.exists():
                ok = load_profile(self.tracker)
                if ok:
                    print("[ColorWar] Loaded calibration profile:", _PROFILE_PATH)
        except Exception:
            pass  # calibration_app not available or profile missing — no problem

    # ------------------------------------------------------------------
    # Sound helpers
    # ------------------------------------------------------------------
    def _play_levelup_sound(self):
        if not self._sound_ready:
            return

        def _play():
            try:
                import pygame
                import numpy as _np
                sr = 44100
                duration = 0.25
                for freq in (523, 784):
                    t = _np.linspace(0, duration, int(sr * duration), endpoint=False)
                    wave = (_np.sin(2 * _np.pi * freq * t) * 16000).astype(_np.int16)
                    sound = pygame.mixer.Sound(buffer=wave.tobytes())
                    sound.play()
                    pygame.time.wait(int(duration * 1000))
            except Exception:
                pass

        threading.Thread(target=_play, daemon=True).start()

    def _play_warning_sound(self):
        if not self._sound_ready:
            return

        def _play():
            try:
                import pygame
                import numpy as _np
                sr = 44100
                duration = 0.15
                freq = 220
                t = _np.linspace(0, duration, int(sr * duration), endpoint=False)
                wave = (_np.sin(2 * _np.pi * freq * t) * 12000).astype(_np.int16)
                sound = pygame.mixer.Sound(buffer=wave.tobytes())
                sound.play()
            except Exception:
                pass

        threading.Thread(target=_play, daemon=True).start()

    def _play_progress_beep(self, freq: int):
        if not self._sound_ready:
            return

        def _play():
            try:
                import pygame
                import numpy as _np
                sr = 44100
                duration = 0.08
                t = _np.linspace(0, duration, int(sr * duration), endpoint=False)
                env = _np.ones_like(t)
                fade = int(sr * 0.01)
                env[:fade] = _np.linspace(0, 1, fade)
                env[-fade:] = _np.linspace(1, 0, fade)
                wave = (_np.sin(2 * _np.pi * freq * t) * env * 10000).astype(_np.int16)
                sound = pygame.mixer.Sound(buffer=wave.tobytes())
                sound.play()
            except Exception:
                pass

        threading.Thread(target=_play, daemon=True).start()

    def _play_ending_jingle(self):
        if not self._sound_ready:
            return

        def _play():
            try:
                import pygame
                import numpy as _np
                sr = 44100
                note_dur = 0.1
                for freq in (880, 698, 523):
                    t = _np.linspace(0, note_dur, int(sr * note_dur), endpoint=False)
                    env = _np.ones_like(t)
                    fade = int(sr * 0.01)
                    env[:fade] = _np.linspace(0, 1, fade)
                    env[-fade:] = _np.linspace(1, 0, fade)
                    wave = (_np.sin(2 * _np.pi * freq * t) * env * 14000).astype(_np.int16)
                    sound = pygame.mixer.Sound(buffer=wave.tobytes())
                    sound.play()
                    pygame.time.wait(int(note_dur * 1000))
            except Exception:
                pass

        threading.Thread(target=_play, daemon=True).start()

    def _play_hit_sound(self):
        if not self._sound_ready:
            return

        def _play():
            try:
                import pygame
                import numpy as _np
                sr = 44100
                duration = 0.06
                freq = 600 + random.randint(-50, 50)
                t = _np.linspace(0, duration, int(sr * duration), endpoint=False)
                env = _np.exp(-t * 30)
                wave = (_np.sin(2 * _np.pi * freq * t) * env * 14000).astype(_np.int16)
                sound = pygame.mixer.Sound(buffer=wave.tobytes())
                sound.play()
            except Exception:
                pass

        threading.Thread(target=_play, daemon=True).start()

    def _play_combo_sound(self, combo_count: int):
        if not self._sound_ready:
            return

        def _play():
            try:
                import pygame
                import numpy as _np
                sr = 44100
                duration = 0.12
                base = 500 + combo_count * 60
                for freq in (base, base + 200):
                    t = _np.linspace(0, duration, int(sr * duration), endpoint=False)
                    env = _np.ones_like(t)
                    fade = int(sr * 0.015)
                    env[:fade] = _np.linspace(0, 1, fade)
                    env[-fade:] = _np.linspace(1, 0, fade)
                    wave = (_np.sin(2 * _np.pi * freq * t) * env * 14000).astype(_np.int16)
                    sound = pygame.mixer.Sound(buffer=wave.tobytes())
                    sound.play()
                    pygame.time.wait(int(duration * 500))
            except Exception:
                pass

        threading.Thread(target=_play, daemon=True).start()

    def _play_lock_sound(self):
        """Play a happy ascending chime when a paddle locks."""
        if not self._sound_ready:
            return

        def _play():
            try:
                import pygame
                import numpy as _np
                sr = 44100
                for freq in (600, 800, 1000):
                    dur = 0.1
                    t = _np.linspace(0, dur, int(sr * dur), endpoint=False)
                    env = _np.ones_like(t)
                    fade = int(sr * 0.01)
                    env[:fade] = _np.linspace(0, 1, fade)
                    env[-fade:] = _np.linspace(1, 0, fade)
                    wave = (_np.sin(2 * _np.pi * freq * t) * env * 14000).astype(_np.int16)
                    sound = pygame.mixer.Sound(buffer=wave.tobytes())
                    sound.play()
                    pygame.time.wait(int(dur * 800))
            except Exception:
                pass

        threading.Thread(target=_play, daemon=True).start()

    # ------------------------------------------------------------------
    # Screen shake helpers
    # ------------------------------------------------------------------
    def _trigger_shake(self, intensity: int):
        self._shake_until = time.time() + SHAKE_DURATION
        self._shake_intensity = max(self._shake_intensity, intensity)

    def _apply_shake(self, frame: np.ndarray) -> np.ndarray:
        now = time.time()
        if now >= self._shake_until:
            self._shake_intensity = 0
            return frame
        remaining = (self._shake_until - now) / SHAKE_DURATION
        mag = int(self._shake_intensity * remaining)
        if mag < 1:
            return frame
        dx = random.randint(-mag, mag)
        dy = random.randint(-mag, mag)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                              borderMode=cv2.BORDER_REPLICATE)

    # ------------------------------------------------------------------
    # Audio scheduler
    # ------------------------------------------------------------------
    def _update_audio_scheduler(self):
        now = time.time()
        tl = self.engine.time_left

        if now >= self._next_progress_beep and tl > ENDING_WARNING_SECONDS:
            ratio = 1.0 - (tl / GAME_DURATION)
            freq = int(PROGRESS_BEEP_MIN_FREQ +
                       ratio * (PROGRESS_BEEP_MAX_FREQ - PROGRESS_BEEP_MIN_FREQ))
            self._play_progress_beep(freq)
            self._next_progress_beep = now + PROGRESS_BEEP_INTERVAL

        if tl <= ENDING_WARNING_SECONDS and tl > 0:
            if not self._ending_jingle_active:
                self._ending_jingle_active = True
                self._next_ending_jingle = now
            if now >= self._next_ending_jingle:
                self._play_ending_jingle()
                self._next_ending_jingle = now + ENDING_JINGLE_INTERVAL
        else:
            self._ending_jingle_active = False

    # ------------------------------------------------------------------
    # Game start helper
    # ------------------------------------------------------------------
    def _start_game(self):
        diff = self.ui.selected_difficulty
        self.challenges.configure(diff)
        self.engine.reset(self.challenges.speed_multiplier)
        self.state = STATE_GAME
        self.ui.reset_hover_states()
        self._help_show_until = time.time() + 4.0
        now = time.time()
        self._next_progress_beep = now + PROGRESS_BEEP_INTERVAL
        self._next_ending_jingle = 0.0
        self._ending_jingle_active = False
        self._shake_until = 0.0
        self._shake_intensity = 0
        self._replay_frames = []
        self._game_frame_idx = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        cv2.namedWindow("Color War Pong", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Color War Pong", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Track which players were already locked to detect fresh lock events
        _prev_p1_locked = False
        _prev_p2_locked = False

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_blurred = self.tracker.blur_hsv(hsv)

            # Provide raw BGR frame for edge/motion detection
            self.tracker.begin_frame(frame)

            # --- Auto-detect / calibration phase ---
            if self.state == STATE_AUTODETECT:
                self.tracker.calibrate_step(hsv, hsv_blurred)

                # Play lock sound on fresh locks
                if self.tracker.p1_locked and not _prev_p1_locked:
                    self._play_lock_sound()
                if self.tracker.p2_locked and not _prev_p2_locked:
                    self._play_lock_sound()
                _prev_p1_locked = self.tracker.p1_locked
                _prev_p2_locked = self.tracker.p2_locked

                final = self._handle_autodetect(frame, hsv_blurred)

            else:
                # --- Normal tracking (home, game, pause, results, replay) ---
                self.p1_prev = self.p1_pos.copy()
                self.p2_prev = self.p2_pos.copy()
                pos1, cnt1 = self.tracker.track_paddle(hsv_blurred, is_p1=True)
                pos2, cnt2 = self.tracker.track_paddle(hsv_blurred, is_p1=False)
                self.p1_pos = pos1.astype(float)
                self.p2_pos = pos2.astype(float)
                self.p1_contour = cnt1
                self.p2_contour = cnt2

                # Adaptive HSV during gameplay
                self.tracker.adaptive_hsv_update(hsv, cnt1, is_p1=True)
                self.tracker.adaptive_hsv_update(hsv, cnt2, is_p1=False)

                if self.state == STATE_HOME:
                    final = self._handle_home(frame)
                elif self.state == STATE_GAME:
                    final = self._handle_game(frame)
                elif self.state == STATE_PAUSE:
                    final = self._handle_pause(frame)
                elif self.state == STATE_RESULTS:
                    final = self._handle_results(frame)
                elif self.state == STATE_REPLAY:
                    final = self._handle_replay()
                else:
                    final = frame

            # End-of-frame bookkeeping
            self.tracker.end_frame()

            cv2.imshow("Color War Pong", final)
            self.tracker.show_debug_masks(hsv)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c") and self.state in (STATE_HOME, STATE_RESULTS):
                # Quick re-enter calibration from home/results for fast iteration
                self.tracker = PaddleTracker()
                self._load_calibration_profile()
                self.state = STATE_AUTODETECT
                _prev_p1_locked = False
                _prev_p2_locked = False

        self.tracker.close()
        self.cap.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Midline helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp_p1(pos):
        clamped = pos.copy()
        clamped[0] = min(clamped[0], WIDTH // 2 - PADDLE_RADIUS)
        return clamped

    @staticmethod
    def _clamp_p2(pos):
        clamped = pos.copy()
        clamped[0] = max(clamped[0], WIDTH // 2 + PADDLE_RADIUS)
        return clamped

    @staticmethod
    def _is_p1_wrong_side(pos) -> bool:
        return pos[0] > WIDTH // 2

    @staticmethod
    def _is_p2_wrong_side(pos) -> bool:
        return pos[0] < WIDTH // 2

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------
    def _handle_autodetect(self, frame: np.ndarray, hsv_blurred: np.ndarray) -> np.ndarray:
        """Draw calibration overlay and transition to home when both paddles are locked."""
        self.ui.draw_calibration(
            frame,
            self.tracker.calibration_progress(True),
            self.tracker.calibration_progress(False),
            self.tracker.p1_locked,
            self.tracker.p2_locked,
        )

        if self.tracker.both_locked:
            # Set fixed player colours
            self.ui.p1_color = COLOR_P1
            self.ui.p2_color = COLOR_P2
            self.engine.set_player_colors(COLOR_P1, COLOR_P2, "CYAN", "PINK")
            self.state = STATE_HOME
            self.ui.reset_hover_states()

        return frame

    def _handle_home(self, frame: np.ndarray) -> np.ndarray:
        action = self.ui.draw_home(frame, self.p1_pos, self.p2_pos)
        if action == "start":
            self._start_game()
        return frame

    def _handle_game(self, frame: np.ndarray) -> np.ndarray:
        in_prestart = self.engine.in_prestart

        # Record raw camera frame for timelapse (skip prestart)
        if not in_prestart:
            self._game_frame_idx += 1
            if self._game_frame_idx % REPLAY_SAMPLE_EVERY == 0:
                self._replay_frames.append(frame.copy())

        # Determine effective (game-logic) paddle positions
        if in_prestart:
            game_p1 = self.p1_pos.copy()
            game_p2 = self.p2_pos.copy()

            p1_wrong = self._is_p1_wrong_side(self.p1_pos)
            p2_wrong = self._is_p2_wrong_side(self.p2_pos)

            if p1_wrong and not self._p1_was_wrong_side:
                self._play_warning_sound()
            if p2_wrong and not self._p2_was_wrong_side:
                self._play_warning_sound()
            self._p1_was_wrong_side = p1_wrong
            self._p2_was_wrong_side = p2_wrong
        else:
            game_p1 = self._clamp_p1(self.p1_pos)
            game_p2 = self._clamp_p2(self.p2_pos)
            self._p1_was_wrong_side = False
            self._p2_was_wrong_side = False

        # Pause gesture check — require BOTH paddles at decent confidence
        p1_conf = self.tracker.get_confidence(True)
        p2_conf = self.tracker.get_confidence(False)
        pause_trusted = (p1_conf >= TRACKING_CONFIDENCE_MIN and
                         p2_conf >= TRACKING_CONFIDENCE_MIN)
        if pause_trusted and self.ui.check_pause_gesture(self.p1_pos, self.p2_pos):
            self._pause_start = time.time()
            self.state = STATE_PAUSE
            self.ui.reset_hover_states()
            return self._compose_game_frame(frame, game_p1, game_p2)

        # Contours for collision — only pass trusted contours
        p1_cnt = (self.p1_contour
                  if self.p1_contour is not None
                  and not self._is_p1_wrong_side(self.p1_pos)
                  and p1_conf >= TRACKING_CONFIDENCE_MIN
                  else None)
        p2_cnt = (self.p2_contour
                  if self.p2_contour is not None
                  and not self._is_p2_wrong_side(self.p2_pos)
                  and p2_conf >= TRACKING_CONFIDENCE_MIN
                  else None)

        obstacles = self.challenges.obstacles if self.challenges.obstacles_enabled else None
        self.challenges.update(self.engine)
        still_running = self.engine.update(
            game_p1, game_p2, self.p1_prev, self.p2_prev, obstacles,
            p1_contour=p1_cnt, p2_contour=p2_cnt,
        )

        # Consume event flags
        if self.engine.evt_level_up:
            self._play_levelup_sound()
            self._trigger_shake(SHAKE_INTENSITY_LEVEL)

        if self.engine.evt_hit:
            self._play_hit_sound()
            self._trigger_shake(SHAKE_INTENSITY_HIT)
            if (self.engine.combo_count >= 3 and
                    self.engine.combo_banner_until > time.time()):
                self._play_combo_sound(self.engine.combo_count)

        self._update_audio_scheduler()

        if not still_running:
            self.state = STATE_RESULTS
            self.ui.reset_hover_states()

        return self._compose_game_frame(frame, game_p1, game_p2)

    def _handle_pause(self, frame: np.ndarray) -> np.ndarray:
        game_p1 = self._clamp_p1(self.p1_pos)
        game_p2 = self._clamp_p2(self.p2_pos)
        game_frame = self._compose_game_frame(frame, game_p1, game_p2)
        action = self.ui.draw_pause(game_frame, self.p1_pos, self.p2_pos)

        if action == "resume":
            paused = time.time() - self._pause_start
            self.engine.start_time += paused
            self.engine.last_powerup_time += paused
            if self.engine.mega_brush_timer > 0:
                self.engine.mega_brush_timer += paused
            self.challenges._next_ball_spawn += paused
            self.challenges._next_bomb_spawn += paused
            self.challenges._next_obstacle_refresh += paused
            self.engine._next_level_time += paused
            self.engine._next_ball_add_time += paused
            if self.engine.prestart_until > 0:
                self.engine.prestart_until += paused
            self._next_progress_beep += paused
            if self._next_ending_jingle > 0:
                self._next_ending_jingle += paused
            if self.engine.combo_banner_until > 0:
                self.engine.combo_banner_until += paused
            if self.engine._combo_last_hit > 0:
                self.engine._combo_last_hit += paused
            if self._shake_until > 0:
                self._shake_until += paused
            self.state = STATE_GAME
            self.ui.reset_hover_states()
        elif action == "restart":
            self._start_game()
        elif action == "home":
            self.state = STATE_HOME
            self.ui.reset_hover_states()

        return game_frame

    def _handle_results(self, frame: np.ndarray) -> np.ndarray:
        action, res_frame = self.ui.draw_results(
            frame,
            self.engine.paint_canvas,
            self.engine.winner_text,
            self.engine.p1_score_pct,
            self.engine.p2_score_pct,
            self.p1_pos,
            self.p2_pos,
        )
        if action == "replay":
            self._start_replay()
        elif action == "play_again":
            self._start_game()
        elif action == "home":
            self.state = STATE_HOME
            self.ui.reset_hover_states()
        return res_frame

    # ------------------------------------------------------------------
    # Timelapse replay
    # ------------------------------------------------------------------
    def _start_replay(self):
        self.state = STATE_REPLAY
        self.ui.reset_hover_states()
        self._replay_playback_idx = 0.0
        total = len(self._replay_frames)
        if total == 0:
            self.state = STATE_RESULTS
            return
        est_fps = 30.0
        self._replay_step = total / (REPLAY_DURATION * est_fps)

    def _handle_replay(self) -> np.ndarray:
        total = len(self._replay_frames)
        idx = min(int(self._replay_playback_idx), total - 1)
        replay_frame = self._replay_frames[idx].copy()

        # Progress bar
        progress = (idx + 1) / total
        bar_h = 8
        bar_y = HEIGHT - bar_h - 4
        cv2.rectangle(replay_frame, (10, bar_y), (WIDTH - 10, bar_y + bar_h),
                       COLOR_GRAY, -1)
        fill_w = int((WIDTH - 20) * progress)
        cv2.rectangle(replay_frame, (10, bar_y), (10 + fill_w, bar_y + bar_h),
                       COLOR_WHITE, -1)

        cv2.putText(replay_frame, "REPLAY", (15, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        speed_label = f"{total / (REPLAY_DURATION * 30):.1f}x"
        cv2.putText(replay_frame, speed_label, (WIDTH - 80, 30),
                     cv2.FONT_HERSHEY_PLAIN, 1.2, COLOR_WHITE, 1)

        self._replay_playback_idx += self._replay_step

        if self._replay_playback_idx >= total:
            self.state = STATE_RESULTS
            self.ui.reset_hover_states()

        return replay_frame

    # ------------------------------------------------------------------
    # Game-frame compositor
    # ------------------------------------------------------------------
    def _compose_game_frame(self, frame: np.ndarray,
                            game_p1=None, game_p2=None) -> np.ndarray:
        if game_p1 is None:
            game_p1 = self.p1_pos
        if game_p2 is None:
            game_p2 = self.p2_pos

        blend = cv2.addWeighted(frame, 0.6, self.engine.paint_canvas, 0.4, 0)

        # Obstacles
        self.challenges.draw_obstacles(blend)

        # Powerup
        if self.engine.powerup_active:
            px = int(self.engine.powerup_pos[0])
            py = int(self.engine.powerup_pos[1])
            pulse = abs(np.sin(time.time() * 5)) * 10
            cv2.circle(blend, (px, py), int(20 + pulse), COLOR_POWERUP, 2)
            cv2.circle(blend, (px, py), 15, COLOR_POWERUP, -1)
            cv2.putText(blend, "?", (px - 10, py + 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_BLACK, 2)

        # Midline
        cv2.line(blend, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), COLOR_MIDLINE, 1)

        # Mega brush indicator
        if time.time() < self.engine.mega_brush_timer:
            cv2.putText(blend, "MEGA BRUSH!", (WIDTH // 2 - 80, 80),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        in_prestart = self.engine.in_prestart
        p1_wrong = self._is_p1_wrong_side(self.p1_pos)
        p2_wrong = self._is_p2_wrong_side(self.p2_pos)

        # Paddles — colour reflects confidence level
        p1_conf = self.tracker.get_confidence(True)
        p2_conf = self.tracker.get_confidence(False)
        p1_col = self.engine.p1_color if not (in_prestart and p1_wrong) else COLOR_GRAY
        p2_col = self.engine.p2_color if not (in_prestart and p2_wrong) else COLOR_GRAY

        # Dim paddle contour colour when confidence is low (visual cue)
        def _dim(color, conf):
            if conf >= TRACKING_CONFIDENCE_MIN:
                return color
            # Blend toward gray proportional to lack of confidence
            t = max(0.0, conf / TRACKING_CONFIDENCE_MIN)
            return tuple(int(c * t + 100 * (1 - t)) for c in color)

        p1_draw_col = _dim(p1_col, p1_conf)
        p2_draw_col = _dim(p2_col, p2_conf)

        if self.p1_contour is not None and not (in_prestart and p1_wrong):
            thickness = 3 if p1_conf >= TRACKING_CONFIDENCE_MIN else 1
            cv2.drawContours(blend, [self.p1_contour], -1, p1_draw_col, thickness)
        else:
            cv2.circle(blend, (int(game_p1[0]), int(game_p1[1])), PADDLE_RADIUS, p1_draw_col, 3)

        if self.p2_contour is not None and not (in_prestart and p2_wrong):
            thickness = 3 if p2_conf >= TRACKING_CONFIDENCE_MIN else 1
            cv2.drawContours(blend, [self.p2_contour], -1, p2_draw_col, thickness)
        else:
            cv2.circle(blend, (int(game_p2[0]), int(game_p2[1])), PADDLE_RADIUS, p2_draw_col, 3)

        # Ghost circles when on wrong side (post-prestart)
        if not in_prestart:
            if p1_wrong:
                cv2.circle(blend, (int(self.p1_pos[0]), int(self.p1_pos[1])),
                           PADDLE_RADIUS, COLOR_GRAY, 2)
            if p2_wrong:
                cv2.circle(blend, (int(self.p2_pos[0]), int(self.p2_pos[1])),
                           PADDLE_RADIUS, COLOR_GRAY, 2)

        # Balls
        for ball in self.engine.balls:
            self.challenges.draw_ball(blend, ball,
                                      self.engine.p1_color, self.engine.p2_color)

        # Particles
        for p in self.engine.particles:
            cv2.circle(blend, (int(p[0]), int(p[1])), max(1, int(p[5] / 4)), p[4], -1)

        # Fog (disabled but kept for interface consistency)
        blend = self.challenges.apply_fog(blend)

        # HUD
        self.ui.draw_hud(blend, self.engine.time_left, self.ui.selected_difficulty,
                         self.engine.level, self.engine.level_up_until,
                         combo_count=self.engine.combo_count,
                         combo_owner=self.engine.combo_owner,
                         combo_banner_until=self.engine.combo_banner_until,
                         p1_color=self.engine.p1_color,
                         p2_color=self.engine.p2_color)

        # Pause gesture progress
        if self.ui.pause_hold_start is not None:
            prog = (time.time() - self.ui.pause_hold_start) / PAUSE_HOLD_DURATION
            cv2.putText(blend, f"Pausing... {int(prog * 100)}%",
                         (WIDTH // 2 - 70, HEIGHT - 20),
                         cv2.FONT_HERSHEY_PLAIN, 1.2, COLOR_WHITE, 1)

        # Pre-start countdown overlay
        now = time.time()
        if in_prestart:
            remaining = max(0, self.engine.prestart_until - now)
            overlay = blend.copy()
            cv2.rectangle(overlay, (WIDTH // 2 - 130, HEIGHT // 2 - 40),
                           (WIDTH // 2 + 130, HEIGHT // 2 + 50), COLOR_BLACK, -1)
            cv2.addWeighted(overlay, 0.65, blend, 0.35, 0, blend)
            cv2.putText(blend, f"GET READY  {int(remaining) + 1}",
                         (WIDTH // 2 - 110, HEIGHT // 2),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)
            cv2.putText(blend, "Stay on your side!",
                         (WIDTH // 2 - 90, HEIGHT // 2 + 35),
                         cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 200, 255), 1)

        elif now < self._help_show_until:
            fade = min(1.0, (self._help_show_until - now) / 2.0)
            overlay = blend.copy()
            cv2.rectangle(overlay, (WIDTH // 2 - 160, HEIGHT // 2 - 60),
                           (WIDTH // 2 + 160, HEIGHT // 2 + 70), COLOR_BLACK, -1)
            cv2.addWeighted(overlay, 0.7 * fade, blend, 1.0 - 0.7 * fade, 0, blend)
            alpha_col = tuple(int(c * fade) for c in COLOR_WHITE)
            hints = [
                "Hit the ball to paint the arena!",
                "Hold BOTH paddles at TOP to pause",
                "Most paint coverage wins",
                "Q = Quit",
            ]
            for i, txt in enumerate(hints):
                cv2.putText(blend, txt,
                             (WIDTH // 2 - 145, HEIGHT // 2 - 35 + i * 28),
                             cv2.FONT_HERSHEY_PLAIN, 1.1, alpha_col, 1)

        # Screen shake
        blend = self._apply_shake(blend)

        return blend


# ---------------------------------------------------------------------------
# Module-level entry point
# ---------------------------------------------------------------------------
def main():
    app = ColorWarApp()
    app.run()


if __name__ == "__main__":
    main()
