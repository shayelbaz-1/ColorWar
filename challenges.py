"""Challenge manager: obstacles, fog, bombs, transparency, ball spawning."""

import cv2
import numpy as np
import time
import random

from .config import (
    WIDTH,
    HEIGHT,
    BALL_RADIUS,
    DIFFICULTY_CONFIG,
    COLOR_P1,
    COLOR_P2,
    COLOR_WHITE,
    COLOR_BLACK,
    OBSTACLE_COLOR,
    OBSTACLE_THICKNESS,
    OBSTACLES_PER_SIDE,
    OBSTACLE_REFRESH_SECONDS,
)


class ChallengeManager:
    """Manages difficulty-driven challenges (obstacles, fog, bombs, etc.)."""

    def __init__(self):
        self.obstacles_enabled: bool = False
        self.fog_enabled: bool = False
        self.transparency_enabled: bool = False
        self.bomb_enabled: bool = False
        self.multi_ball: bool = False
        self.speed_multiplier: float = 1.0
        self.fog_side: str = "left"

        self.obstacles: list[tuple[int, int, int, int]] = []
        self._fog_phase: float = 0.0
        self._next_ball_spawn: float = 0.0
        self._next_bomb_spawn: float = 0.0
        self._next_obstacle_refresh: float = 0.0

    def configure(self, difficulty: int):
        """Apply difficulty settings."""
        cfg = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG[0])
        self.speed_multiplier = cfg.get("speed_mult", 1.0)
        self.multi_ball = cfg.get("multi_ball", False)
        self.fog_enabled = cfg.get("fog", False)
        self.transparency_enabled = cfg.get("transparency", False)
        self.obstacles_enabled = cfg.get("obstacles", False)
        self.bomb_enabled = cfg.get("bomb", False)

        self.obstacles = []
        now = time.time()
        if self.obstacles_enabled:
            self._generate_obstacles()
            self._next_obstacle_refresh = now + OBSTACLE_REFRESH_SECONDS

        self._next_ball_spawn = now + random.uniform(8, 15)
        self._next_bomb_spawn = now + random.uniform(15, 25)
        self.fog_side = random.choice(["left", "right"])
        self._fog_phase = 0.0

    def _generate_obstacles(self):
        """Create balanced obstacle rectangles: equal count and size range per side."""
        self.obstacles = []
        mid = WIDTH // 2
        for _ in range(OBSTACLES_PER_SIDE):
            w = random.randint(40, 80)
            h = random.randint(8, 15)
            y = random.randint(80, HEIGHT - 80 - h)
            # Left half
            x_left = random.randint(60, mid - 30 - w)
            self.obstacles.append((x_left, y, w, h))
            # Right half (mirror-ish placement)
            x_right = random.randint(mid + 30, WIDTH - 60 - w)
            y_right = random.randint(80, HEIGHT - 80 - h)
            self.obstacles.append((x_right, y_right, w, h))

    def update(self, game_engine):
        """Per-frame update: spawning, transparency, obstacle refresh."""
        now = time.time()

        # Spawn extra ball if multi_ball enabled
        if self.multi_ball and now >= self._next_ball_spawn:
            game_engine.add_ball("normal")
            self._next_ball_spawn = now + random.uniform(12, 20)

        # Spawn bomb ball
        if self.bomb_enabled and now >= self._next_bomb_spawn:
            game_engine.add_ball("bomb")
            self._next_bomb_spawn = now + random.uniform(15, 30)

        # Transparency
        for ball in game_engine.balls:
            ball.alpha = 0.35 if (self.transparency_enabled and ball.ball_type == "normal") else 1.0

        # Refresh obstacles on timer
        if self.obstacles_enabled and now >= self._next_obstacle_refresh:
            self._generate_obstacles()
            self._next_obstacle_refresh = now + OBSTACLE_REFRESH_SECONDS

    def apply_fog(self, frame: np.ndarray) -> np.ndarray:
        """Fog effect has been disabled. Returns the frame unchanged."""
        return frame

    def draw_ball(self, frame: np.ndarray, ball,
                  p1_color=COLOR_P1, p2_color=COLOR_P2):
        """Draw a single ball, handling bomb and transparency variants."""
        bx, by = int(ball.pos[0]), int(ball.pos[1])

        if ball.ball_type == "bomb":
            pulse = abs(np.sin(time.time() * 8))
            color = (0, 0, int(150 + 105 * pulse))
            cv2.circle(frame, (bx, by), BALL_RADIUS, color, -1)
            cv2.circle(frame, (bx, by), BALL_RADIUS, (0, 0, 255), 2)
            cv2.putText(frame, "X", (bx - 7, by + 7),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
            return

        # Normal ball colour from last hitter
        if ball.last_hitter == "P1":
            b_col = p1_color
        elif ball.last_hitter == "P2":
            b_col = p2_color
        else:
            b_col = COLOR_WHITE

        # Colored glowing aura (two layers)
        aura1 = int(BALL_RADIUS * 1.8)
        aura2 = int(BALL_RADIUS * 1.4)
        overlay = frame.copy()
        cv2.circle(overlay, (bx, by), aura1, b_col, -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        
        overlay = frame.copy()
        cv2.circle(overlay, (bx, by), aura2, b_col, -1)
        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

        # Draw main solid body with potential transparency
        if hasattr(ball, "alpha") and ball.alpha < 1.0:
            overlay2 = frame.copy()
            cv2.circle(overlay2, (bx, by), BALL_RADIUS, b_col, -1)
            cv2.addWeighted(overlay2, ball.alpha, frame, 1.0 - ball.alpha, 0, frame)
        else:
            cv2.circle(frame, (bx, by), BALL_RADIUS, b_col, -1)

        # High-contrast border (outer black, inner white) to pop against any background
        cv2.circle(frame, (bx, by), BALL_RADIUS + 1, (0, 0, 0), 2)
        cv2.circle(frame, (bx, by), BALL_RADIUS, (255, 255, 255), 1)

    def draw_obstacles(self, frame: np.ndarray):
        """Draw all active obstacle rectangles."""
        for ox, oy, ow, oh in self.obstacles:
            cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh),
                          OBSTACLE_COLOR, OBSTACLE_THICKNESS)
