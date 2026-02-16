"""Core game engine: ball physics, collisions, trail painting, particles, scoring."""

import cv2
import numpy as np
import time
import random

from .config import (
    WIDTH,
    HEIGHT,
    BALL_RADIUS,
    BALL_INITIAL_SPEED,
    BALL_MIN_SPEED,
    BALL_MAX_SPEED,
    BALL_FRICTION,
    PADDLE_RADIUS,
    GAME_DURATION,
    PRESTART_DURATION,
    POWERUP_SPAWN_TIME,
    MEGA_BRUSH_DURATION,
    COLOR_P1,
    COLOR_P2,
    COLOR_BOMB,
    PARTICLE_COLORS,
    LEVEL_INTERVAL_SECONDS,
    LEVEL_SPEED_MULT,
    BALL_ADD_INTERVAL_SECONDS,
    MAX_BALLS,
    LEVEL_OVERLAY_DURATION,
    PADDLE_BLOB_COLLISION_PADDING,
    COMBO_MIN_HITS,
    COMBO_TIMEOUT,
    COMBO_BANNER_DURATION,
    CONFETTI_COUNT_LEVEL,
    CONFETTI_COUNT_COMBO,
    CONFETTI_COLORS,
)


# ---------------------------------------------------------------------------
# Ball
# ---------------------------------------------------------------------------
class Ball:
    """A single ball (normal or bomb)."""

    def __init__(self, pos, vel, ball_type="normal"):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.ball_type = ball_type       # "normal" | "bomb"
        self.last_hitter = None
        self.alpha = 1.0                 # 1.0 = fully opaque


# ---------------------------------------------------------------------------
# Game engine
# ---------------------------------------------------------------------------
class GameEngine:
    """Physics, collision, painting, particles, powerups, and scoring."""

    def __init__(self):
        self.balls: list[Ball] = []
        self.paint_canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.particles: list = []
        self.start_time: float = 0

        # Dynamic player colors (cyan/pink fixed)
        self.p1_color: tuple = COLOR_P1
        self.p2_color: tuple = COLOR_P2
        self.p1_name: str = "CYAN"
        self.p2_name: str = "PINK"

        # Powerups
        self.powerup_active = False
        self.powerup_pos = np.array([0, 0])
        self.last_powerup_time: float = 0
        self.mega_brush_timer: float = 0

        # Scoring
        self.winner_text = ""
        self.p1_score_pct = 0
        self.p2_score_pct = 0

        # Speed scale
        self.speed_multiplier = 1.0

        # Level progression
        self.level: int = 1
        self._next_level_time: float = 0
        self._next_ball_add_time: float = 0
        self.level_up_until: float = 0       # timestamp until which overlay shows

        # Pre-start idle phase
        self.prestart_until: float = 0       # timestamp when prestart ends

        # Combo tracking
        self.combo_owner: str | None = None  # "P1" or "P2"
        self.combo_count: int = 0
        self._combo_last_hit: float = 0
        self.combo_banner_until: float = 0   # timestamp to show banner

        # Per-frame event flags (consumed & reset each frame by caller)
        self.evt_hit: bool = False           # any paddle hit this frame
        self.evt_level_up: bool = False      # level-up happened this frame

    def set_player_colors(self, p1_color: tuple, p2_color: tuple,
                          p1_name: str = "P1", p2_name: str = "P2"):
        """Set dynamic player colours (BGR tuples) and display names."""
        self.p1_color = p1_color
        self.p2_color = p2_color
        self.p1_name = p1_name
        self.p2_name = p2_name

    # ------------------------------------------------------------------
    # Reset / helpers
    # ------------------------------------------------------------------
    def reset(self, speed_mult: float = 1.0):
        """Prepare a fresh round."""
        self.speed_multiplier = speed_mult
        speed = BALL_INITIAL_SPEED * speed_mult
        self.balls = [
            Ball(
                [WIDTH / 2, HEIGHT / 2],
                [
                    random.choice([-1, 1]) * speed,
                    random.choice([-1, 1]) * (speed - 1),
                ],
            )
        ]
        self.paint_canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.particles = []
        now = time.time()
        self.start_time = now
        self.last_powerup_time = now
        self.powerup_active = False
        self.mega_brush_timer = 0
        self.winner_text = ""
        self.p1_score_pct = 0
        self.p2_score_pct = 0

        # Level progression
        self.level = 1
        self._next_level_time = now + LEVEL_INTERVAL_SECONDS
        self._next_ball_add_time = now + BALL_ADD_INTERVAL_SECONDS
        self.level_up_until = 0

        # Pre-start idle phase
        self.prestart_until = now + PRESTART_DURATION

        # Combo reset
        self.combo_owner = None
        self.combo_count = 0
        self._combo_last_hit = 0
        self.combo_banner_until = 0

    def add_ball(self, ball_type: str = "normal"):
        """Spawn an additional ball from the centre."""
        speed = BALL_INITIAL_SPEED * self.speed_multiplier
        self.balls.append(
            Ball(
                [WIDTH / 2, HEIGHT / 2],
                [
                    random.choice([-1, 1]) * speed,
                    random.choice([-1, 1]) * (speed - 1),
                ],
                ball_type=ball_type,
            )
        )

    def spawn_particles(self, pos, count: int = 15, color=None):
        for _ in range(count):
            angle = random.uniform(0, 6.28)
            spd = random.uniform(2, 10)
            vx = np.cos(angle) * spd
            vy = np.sin(angle) * spd
            c = color if color else random.choice(PARTICLE_COLORS)
            self.particles.append(
                [float(pos[0]), float(pos[1]), vx, vy, c, random.randint(10, 25)]
            )

    def spawn_confetti(self, pos, count: int = 40):
        """Spawn a burst of bright confetti particles."""
        for _ in range(count):
            angle = random.uniform(0, 6.28)
            spd = random.uniform(3, 14)
            vx = np.cos(angle) * spd
            vy = np.sin(angle) * spd
            c = random.choice(CONFETTI_COLORS)
            self.particles.append(
                [float(pos[0]), float(pos[1]), vx, vy, c, random.randint(18, 40)]
            )

    # ------------------------------------------------------------------
    # Main update (one frame)
    # ------------------------------------------------------------------
    def update(self, p1_pos, p2_pos, p1_prev, p2_prev, obstacles=None,
               p1_contour=None, p2_contour=None):
        """Advance game state by one frame.

        Returns ``True`` while the game is running; ``False`` when time is up.
        """
        now = time.time()

        # Reset per-frame event flags
        self.evt_hit = False
        self.evt_level_up = False

        # Decay combo if timed out
        if self.combo_count > 0 and (now - self._combo_last_hit) > COMBO_TIMEOUT:
            self.combo_owner = None
            self.combo_count = 0

        for ball in self.balls[:]:
            self._update_ball_physics(ball)
            self._wall_bounce(ball)
            self._player_collision(ball, p1_pos, p2_pos, p1_prev, p2_prev,
                                   p1_contour, p2_contour)
            if obstacles:
                self._obstacle_collision(ball, obstacles)
            self._paint_trail(ball, now)

        self._update_powerups(now)
        self._update_particles()
        self._update_progression(now)

        if now - self.start_time > GAME_DURATION:
            self.calculate_results()
            return False
        return True

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------
    def _update_ball_physics(self, ball: Ball):
        ball.pos += ball.vel
        ball.vel *= BALL_FRICTION

    def _wall_bounce(self, ball: Ball):
        # Top
        if ball.pos[1] <= BALL_RADIUS:
            ball.pos[1] = BALL_RADIUS + 1
            ball.vel[1] = abs(ball.vel[1])
            self.spawn_particles(ball.pos)
        # Bottom
        elif ball.pos[1] >= HEIGHT - BALL_RADIUS:
            ball.pos[1] = HEIGHT - BALL_RADIUS - 1
            ball.vel[1] = -abs(ball.vel[1])
            self.spawn_particles(ball.pos)
        # Left
        if ball.pos[0] <= BALL_RADIUS:
            ball.pos[0] = BALL_RADIUS + 1
            ball.vel[0] = abs(ball.vel[0])
            self.spawn_particles(ball.pos)
        # Right
        elif ball.pos[0] >= WIDTH - BALL_RADIUS:
            ball.pos[0] = WIDTH - BALL_RADIUS - 1
            ball.vel[0] = -abs(ball.vel[0])
            self.spawn_particles(ball.pos)

    def _player_collision(self, ball: Ball, p1_pos, p2_pos, p1_prev, p2_prev,
                          p1_contour=None, p2_contour=None):
        players = [
            ("P1", p1_pos, p1_prev, p1_contour),
            ("P2", p2_pos, p2_prev, p2_contour),
        ]
        for pid, pos, prev, contour in players:
            hit = False

            if contour is not None:
                # Contour-based collision: the entire paddle colour blob is hittable
                ball_pt = (float(ball.pos[0]), float(ball.pos[1]))
                signed_dist = cv2.pointPolygonTest(contour, ball_pt, True)
                # signed_dist > 0 → inside contour, < 0 → outside
                # Hit when ball overlaps contour (considering ball radius + padding)
                if signed_dist >= -(BALL_RADIUS + PADDLE_BLOB_COLLISION_PADDING):
                    hit = True
            else:
                # Circle fallback when contour is unavailable
                dist = np.linalg.norm(ball.pos - pos)
                if dist < (BALL_RADIUS + PADDLE_RADIUS) and dist > 0:
                    hit = True

            if hit:
                # Bounce direction: always use centre-to-centre normal
                hand_vel = pos - prev
                dist = np.linalg.norm(ball.pos - pos)
                if dist < 1:
                    dist = 1  # avoid division by zero
                normal = (ball.pos - pos) / dist
                speed = np.linalg.norm(ball.vel)
                impact = np.dot(hand_vel, normal)
                new_speed = min(max(speed + impact * 0.5, BALL_MIN_SPEED), BALL_MAX_SPEED)
                ball.vel = normal * new_speed

                # Push ball out of paddle
                overlap = (BALL_RADIUS + PADDLE_RADIUS) - dist
                if overlap > 0:
                    ball.pos += normal * (overlap + 2)
                else:
                    ball.pos += normal * 3  # small nudge when using contour

                # Bomb ball detonation
                if ball.ball_type == "bomb":
                    self._detonate_bomb(ball, pid)
                    if ball in self.balls:
                        self.balls.remove(ball)
                    return

                ball.last_hitter = pid
                self.spawn_particles(ball.pos, count=20)

                # --- Event flag ---
                self.evt_hit = True

                # --- Combo tracking ---
                now_t = time.time()
                if pid == self.combo_owner:
                    self.combo_count += 1
                else:
                    self.combo_owner = pid
                    self.combo_count = 1
                self._combo_last_hit = now_t
                if self.combo_count >= COMBO_MIN_HITS:
                    self.combo_banner_until = now_t + COMBO_BANNER_DURATION
                    # Confetti burst on every +2 hits past threshold
                    if (self.combo_count - COMBO_MIN_HITS) % 2 == 0:
                        self.spawn_confetti(ball.pos, CONFETTI_COUNT_COMBO)

    def _detonate_bomb(self, ball: Ball, hitter_pid: str):
        """Fill a region of the canvas with the *opponent's* colour."""
        opponent_color = self.p2_color if hitter_pid == "P1" else self.p1_color
        cx, cy = int(ball.pos[0]), int(ball.pos[1])
        cv2.circle(self.paint_canvas, (cx, cy), 80, opponent_color, -1)
        self.spawn_particles(ball.pos, count=50, color=COLOR_BOMB)

    def _obstacle_collision(self, ball: Ball, obstacles):
        for ox, oy, ow, oh in obstacles:
            closest_x = max(ox, min(ball.pos[0], ox + ow))
            closest_y = max(oy, min(ball.pos[1], oy + oh))
            dx = ball.pos[0] - closest_x
            dy = ball.pos[1] - closest_y
            dist = np.sqrt(dx * dx + dy * dy)
            if 0 < dist < BALL_RADIUS:
                normal = np.array([dx, dy]) / dist
                ball.vel -= 2 * np.dot(ball.vel, normal) * normal
                ball.pos += normal * (BALL_RADIUS - dist + 1)
                self.spawn_particles(ball.pos, count=5)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def _paint_trail(self, ball: Ball, now: float):
        if ball.last_hitter is None or ball.ball_type == "bomb":
            return
        brush = BALL_RADIUS
        if now < self.mega_brush_timer:
            brush = BALL_RADIUS * 3
        color = self.p1_color if ball.last_hitter == "P1" else self.p2_color
        cv2.circle(
            self.paint_canvas,
            (int(ball.pos[0]), int(ball.pos[1])),
            brush,
            color,
            -1,
        )

    # ------------------------------------------------------------------
    # Level progression
    # ------------------------------------------------------------------
    def _update_progression(self, now: float):
        """Handle speed bumps and ball additions on a timer."""
        leveled = False

        # Speed bump every LEVEL_INTERVAL_SECONDS
        if now >= self._next_level_time:
            self.speed_multiplier *= LEVEL_SPEED_MULT
            # Also nudge existing balls a bit faster
            for ball in self.balls:
                ball.vel *= LEVEL_SPEED_MULT
            self.level += 1
            self._next_level_time = now + LEVEL_INTERVAL_SECONDS
            leveled = True

        # Add an extra ball every BALL_ADD_INTERVAL_SECONDS (up to MAX_BALLS)
        if now >= self._next_ball_add_time:
            normal_count = sum(1 for b in self.balls if b.ball_type == "normal")
            if normal_count < MAX_BALLS:
                self.add_ball("normal")
                if not leveled:
                    leveled = True
            self._next_ball_add_time = now + BALL_ADD_INTERVAL_SECONDS

        if leveled:
            self.level_up_until = now + LEVEL_OVERLAY_DURATION
            self.evt_level_up = True
            # Confetti burst at screen centre on level-up
            self.spawn_confetti(
                np.array([WIDTH / 2, HEIGHT / 2]), CONFETTI_COUNT_LEVEL
            )

    # ------------------------------------------------------------------
    # Powerups
    # ------------------------------------------------------------------
    def _update_powerups(self, now: float):
        if not self.powerup_active and (now - self.last_powerup_time > POWERUP_SPAWN_TIME):
            self.powerup_active = True
            self.powerup_pos = np.array(
                [random.randint(200, WIDTH - 200), random.randint(100, HEIGHT - 100)]
            )

        if self.powerup_active:
            for ball in self.balls:
                dist = np.linalg.norm(ball.pos - self.powerup_pos)
                if dist < (BALL_RADIUS + 25):
                    self.powerup_active = False
                    self.last_powerup_time = now
                    self.mega_brush_timer = now + MEGA_BRUSH_DURATION
                    self.spawn_particles(self.powerup_pos, count=50, color=(0, 255, 0))
                    break

    # ------------------------------------------------------------------
    # Particles
    # ------------------------------------------------------------------
    def _update_particles(self):
        alive = []
        for p in self.particles:
            p[0] += p[2]
            p[1] += p[3]
            p[5] -= 1
            if p[5] > 0:
                alive.append(p)
        self.particles = alive

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def calculate_results(self):
        p1_px = int(np.sum(np.all(self.paint_canvas == self.p1_color, axis=2)))
        p2_px = int(np.sum(np.all(self.paint_canvas == self.p2_color, axis=2)))
        total = p1_px + p2_px + 1
        self.p1_score_pct = int((p1_px / total) * 100)
        self.p2_score_pct = int((p2_px / total) * 100)

        if p1_px > p2_px:
            self.winner_text = f"{self.p1_name} WINS!"
        elif p2_px > p1_px:
            self.winner_text = f"{self.p2_name} WINS!"
        else:
            self.winner_text = "DRAW!"

    @property
    def in_prestart(self) -> bool:
        """True during the initial idle phase before midline enforcement."""
        return time.time() < self.prestart_until

    @property
    def time_left(self) -> int:
        return max(0, int(GAME_DURATION - (time.time() - self.start_time)))
