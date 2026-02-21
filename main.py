"""Main application: camera loop, state machine, frame composition."""

import cv2
import numpy as np
import time
import pygame
import threading
import random
import math

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
        self.tracker.reset_to_priors()  # Start from broad priors for auto-calibration
        self._calib_step: int = 0
        self._current_bgm: str | None = None

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

        # Dynamic side assignment (True = P1 is left, False = P1 is right)
        self._p1_is_left: bool = True

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

    def _set_bgm(self, path: str | None):
        if not self._sound_ready or self._current_bgm == path:
            return
        self._current_bgm = path
        import pygame
        if path is None:
            pygame.mixer.music.fadeout(500)
            return

        import os
        if os.path.exists(path):
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.set_volume(0.3)
                pygame.mixer.music.play(-1)
            except Exception as e:
                print(f"BGM Error ({path}):", e)

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
        # Dynamic side assignment based on current paddle positions
        self._p1_is_left = self.p1_pos[0] <= self.p2_pos[0]
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
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
        pygame.display.set_caption("Color War Pong")
        pygame.mouse.set_visible(False)
        
        try:
            pygame.mixer.init()
        except Exception:
            pass

        clock = pygame.time.Clock()
        self._calib_step = 0
        running = True

        # Helper to convert OpenCV BGR to Pygame Surface
        def _bgr_to_surface(bgr) -> pygame.Surface:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

        while running:
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
                surf = self._handle_autodetect(frame, hsv, hsv_blurred, _bgr_to_surface)


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
                    surf = self._handle_home(frame, _bgr_to_surface)
                elif self.state == STATE_GAME:
                    surf = self._handle_game(frame, _bgr_to_surface)
                elif self.state == STATE_PAUSE:
                    surf = self._handle_pause(frame, _bgr_to_surface)
                elif self.state == STATE_RESULTS:
                    surf = self._handle_results(frame, _bgr_to_surface)
                elif self.state == STATE_REPLAY:
                    surf = self._handle_replay(_bgr_to_surface)
                else:
                    surf = _bgr_to_surface(frame)

            # End-of-frame bookkeeping
            self.tracker.end_frame()

            screen.blit(surf, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_c and self.state in (STATE_HOME, STATE_RESULTS):
                        # Quick re-enter calibration
                        self.tracker = PaddleTracker()
                        self.tracker.reset_to_priors()  # Fresh broad priors
                        self.state = STATE_AUTODETECT
                        self._calib_step = 0
                        

            
            clock.tick(60)

        self.tracker.close()
        self.cap.release()
        pygame.quit()

    # ------------------------------------------------------------------
    # Midline helpers
    # ------------------------------------------------------------------
    def _clamp_p1(self, pos):
        clamped = pos.copy()
        if self._p1_is_left:
            clamped[0] = min(clamped[0], WIDTH // 2 - PADDLE_RADIUS)
        else:
            clamped[0] = max(clamped[0], WIDTH // 2 + PADDLE_RADIUS)
        return clamped

    def _clamp_p2(self, pos):
        clamped = pos.copy()
        if self._p1_is_left:
            clamped[0] = max(clamped[0], WIDTH // 2 + PADDLE_RADIUS)
        else:
            clamped[0] = min(clamped[0], WIDTH // 2 - PADDLE_RADIUS)
        return clamped

    def _is_p1_wrong_side(self, pos) -> bool:
        if self._p1_is_left:
            return pos[0] > WIDTH // 2
        return pos[0] < WIDTH // 2

    def _is_p2_wrong_side(self, pos) -> bool:
        if self._p1_is_left:
            return pos[0] < WIDTH // 2
        return pos[0] > WIDTH // 2

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------
    def _handle_autodetect(self, frame: np.ndarray, hsv: np.ndarray, hsv_blurred: np.ndarray, bgr_to_surface) -> pygame.Surface:
        """Automatic paddle detection using motion + color fusion."""
        # Run auto-calibration every frame
        self.tracker.calibrate_step(hsv, hsv_blurred)

        p1_progress = self.tracker.calibration_progress(is_p1=True)
        p2_progress = self.tracker.calibration_progress(is_p1=False)
        p1_locked = self.tracker.p1_locked
        p2_locked = self.tracker.p2_locked

        # Play lock sound on transitions
        if p1_locked and self._calib_step == 0:
            self._calib_step = 1
            self._play_lock_sound()
        if p2_locked and self._calib_step < 2:
            self._calib_step = 2
            self._play_lock_sound()

        surf = bgr_to_surface(frame)
        self.ui.draw_auto_calibration(surf, p1_progress, p2_progress, p1_locked, p2_locked)

        if p1_locked and p2_locked:
            self.ui.p1_color = COLOR_P1
            self.ui.p2_color = COLOR_P2
            self.engine.set_player_colors(COLOR_P1, COLOR_P2, "CYAN", "PINK")
            self.state = STATE_HOME
            self.ui.reset_hover_states()
            pygame.mouse.set_visible(False)

        return surf

    def _handle_home(self, frame: np.ndarray, bgr_to_surface) -> pygame.Surface:
        self._set_bgm("assets/menu_bgm.wav")
        self.tracker.in_game = False
        surf = bgr_to_surface(frame)
        action = self.ui.draw_home(surf, self.p1_pos, self.p2_pos)
        if action == "start":
            self._start_game()
        return surf

    def _handle_game(self, frame: np.ndarray, bgr_to_surface) -> pygame.Surface:
        # Decide music pacing
        if self.engine.time_left > 0 and self.engine.time_left <= 20:
            self._set_bgm("assets/game_bgm_fast.wav")
        else:
            self._set_bgm("assets/game_bgm.wav")
            
        self.tracker.in_game = True
        
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
            base_bgr = self._compose_game_frame(frame, game_p1, game_p2)
            surf = bgr_to_surface(base_bgr)
            self.ui.draw_hud(surf, self.engine.time_left, self.ui.selected_difficulty,
                             self.engine.level, self.engine.level_up_until,
                             combo_count=self.engine.combo_count, combo_owner=self.engine.combo_owner,
                             combo_banner_until=self.engine.combo_banner_until,
                             p1_color=self.engine.p1_color, p2_color=self.engine.p2_color)
            return surf

        # Contours for collision — only pass trusted contours
        p1_wrong_now = self._is_p1_wrong_side(self.p1_pos)
        p2_wrong_now = self._is_p2_wrong_side(self.p2_pos)
        p1_cnt = (self.p1_contour
                  if self.p1_contour is not None
                  and not p1_wrong_now
                  and p1_conf >= TRACKING_CONFIDENCE_MIN
                  else None)
        p2_cnt = (self.p2_contour
                  if self.p2_contour is not None
                  and not p2_wrong_now
                  and p2_conf >= TRACKING_CONFIDENCE_MIN
                  else None)

        # Move collision position off-screen when on wrong side
        # so the circle fallback in game.py also can't trigger
        OFF_SCREEN = np.array([-9999.0, -9999.0])
        col_p1 = OFF_SCREEN if p1_wrong_now else game_p1
        col_p2 = OFF_SCREEN if p2_wrong_now else game_p2

        obstacles = self.challenges.obstacles if self.challenges.obstacles_enabled else None
        
        self.challenges.update(self.engine)
        still_running = self.engine.update(
            col_p1, col_p2, self.p1_prev, self.p2_prev, obstacles,
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

        base_bgr = self._compose_game_frame(frame, game_p1, game_p2)
        surf = bgr_to_surface(base_bgr)
        self.ui.draw_hud(surf, self.engine.time_left, self.ui.selected_difficulty,
                         self.engine.level, self.engine.level_up_until,
                         combo_count=self.engine.combo_count, combo_owner=self.engine.combo_owner,
                         combo_banner_until=self.engine.combo_banner_until,
                         p1_color=self.engine.p1_color, p2_color=self.engine.p2_color)

        if self.ui.pause_hold_start is not None:
            self.ui.draw_pause_progress(surf)

        # Pre-start countdown overlay and hints
        if in_prestart:
            self.ui.draw_prestart(surf, self.engine.prestart_until)
        elif time.time() < self._help_show_until:
            self.ui.draw_hints(surf, self._help_show_until)

        return surf

    def _handle_pause(self, frame: np.ndarray, bgr_to_surface) -> pygame.Surface:
        self._set_bgm("assets/menu_bgm.wav")
        self.tracker.in_game = False
        game_p1 = self._clamp_p1(self.p1_pos)
        game_p2 = self._clamp_p2(self.p2_pos)
        base_bgr = self._compose_game_frame(frame, game_p1, game_p2)
        surf = bgr_to_surface(base_bgr)
        self.ui.draw_hud(surf, self.engine.time_left, self.ui.selected_difficulty,
                         self.engine.level, self.engine.level_up_until,
                         combo_count=self.engine.combo_count, combo_owner=self.engine.combo_owner,
                         combo_banner_until=self.engine.combo_banner_until,
                         p1_color=self.engine.p1_color, p2_color=self.engine.p2_color)

        action = self.ui.draw_pause(surf, self.p1_pos, self.p2_pos)

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

        return surf

    def _handle_results(self, frame: np.ndarray, bgr_to_surface) -> pygame.Surface:
        self._set_bgm("assets/menu_bgm.wav")
        self.tracker.in_game = False
        res_bgr = cv2.addWeighted(frame, 0.3, self.engine.paint_canvas, 0.7, 0)
        surf = bgr_to_surface(res_bgr)
        action = self.ui.draw_results(
            surf,
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
        return surf

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

    def _handle_replay(self, bgr_to_surface) -> pygame.Surface:
        total = len(self._replay_frames)
        idx = min(int(self._replay_playback_idx), total - 1)
        replay_frame = self._replay_frames[idx].copy()
        surf = bgr_to_surface(replay_frame)
        self.ui.draw_replay_hud(surf, idx, total)

        self._replay_playback_idx += self._replay_step

        if self._replay_playback_idx >= total:
            self.state = STATE_RESULTS
            self.ui.reset_hover_states()

        return surf

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

        if self.p2_contour is not None and not (in_prestart and p2_wrong):
            thickness = 3 if p2_conf >= TRACKING_CONFIDENCE_MIN else 1
            cv2.drawContours(blend, [self.p2_contour], -1, p2_draw_col, thickness)

        # Pulsating wrong-side warning overlay
        if not in_prestart:
            pulse = abs(math.sin(time.time() * 5))  # 0..1 pulsating
            for is_wrong, player_pos in [(p1_wrong, self.p1_pos), (p2_wrong, self.p2_pos)]:
                if not is_wrong:
                    continue
                # Draw warning on the player's OWN/CORRECT side (to alert them to come back)
                px = int(player_pos[0])
                on_left_half = px >= WIDTH // 2  # player IS on right → their side is LEFT
                # Red/orange glow overlay on the affected half
                overlay = blend.copy()
                alpha = 0.08 + 0.12 * pulse  # subtle pulsating transparency
                if on_left_half:
                    cv2.rectangle(overlay, (0, 0), (WIDTH // 2, HEIGHT), (0, 0, 200), -1)
                    text_cx = WIDTH // 4
                else:
                    cv2.rectangle(overlay, (WIDTH // 2, 0), (WIDTH, HEIGHT), (0, 0, 200), -1)
                    text_cx = 3 * WIDTH // 4
                cv2.addWeighted(overlay, alpha, blend, 1.0 - alpha, 0, blend)
                # Pulsating border stripe
                bx = WIDTH // 2 - 2 if on_left_half else WIDTH // 2
                border_alpha = 0.3 + 0.5 * pulse
                border_overlay = blend.copy()
                cv2.rectangle(border_overlay, (bx, 0), (bx + 4, HEIGHT), (0, 80, 255), -1)
                cv2.addWeighted(border_overlay, border_alpha, blend, 1.0 - border_alpha, 0, blend)
                # "WRONG SIDE!" text with glow
                txt = "WRONG SIDE!"
                font_scale = 0.55 + 0.1 * pulse
                thickness = 2
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                tx = text_cx - tw // 2
                ty = HEIGHT // 2 + th // 2
                # Glow behind text
                for gx, gy in [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, -2), (0, 2), (-2, 0), (2, 0)]:
                    cv2.putText(blend, txt, (tx + gx, ty + gy),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 100), thickness + 1)
                # Main text
                glow_intensity = int(200 + 55 * pulse)
                cv2.putText(blend, txt, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, glow_intensity // 4, glow_intensity), thickness)

        # Balls
        for ball in self.engine.balls:
            self.challenges.draw_ball(blend, ball,
                                      self.engine.p1_color, self.engine.p2_color)

        # Particles (Glowing Orbs)
        for p in self.engine.particles:
            r = max(1, int(p[5] / 3))
            c = p[4]
            cv2.circle(blend, (int(p[0]), int(p[1])), r + 2, c, -1)
            cv2.circle(blend, (int(p[0]), int(p[1])), max(1, r - 1), (255, 255, 255), -1)

        # Fog (disabled but kept for interface consistency)
        blend = self.challenges.apply_fog(blend)

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
