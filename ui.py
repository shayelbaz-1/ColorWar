"""UI layer: home menu, pause menu, results screen, HUD, hover/hold buttons."""

import numpy as np
import time
import pygame
import math
import os

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

pygame.font.init()
font_path = "assets/PressStart2P.ttf"
if os.path.exists(font_path):
    FONT_TITLE = pygame.font.Font(font_path, 40)
    FONT_LARGE = pygame.font.Font(font_path, 24)
    FONT_MEDIUM = pygame.font.Font(font_path, 16)
    FONT_SMALL = pygame.font.Font(font_path, 10)
else:
    try:
        FONT_TITLE = pygame.font.SysFont("courier", 50, bold=True)
        FONT_LARGE = pygame.font.SysFont("courier", 36, bold=True)
        FONT_MEDIUM = pygame.font.SysFont("courier", 24, bold=True)
        FONT_SMALL = pygame.font.SysFont("courier", 16)
    except:
        FONT_TITLE = pygame.font.Font(None, 60)
        FONT_LARGE = pygame.font.Font(None, 40)
        FONT_MEDIUM = pygame.font.Font(None, 24)
        FONT_SMALL = pygame.font.Font(None, 18)

def to_rgb(bgr_color):
    """Convert OpenCV BGR to Pygame RGB."""
    if isinstance(bgr_color, (list, tuple)):
        return (bgr_color[2], bgr_color[1], bgr_color[0])
    return bgr_color

# ---------------------------------------------------------------------------
# Hover button
# ---------------------------------------------------------------------------
class HoverButton:
    """A UI button activated by holding a paddle over it for HOVER_DURATION."""

    def __init__(self, x: int, y: int, w: int, h: int, label: str, color=(0, 255, 0)):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.color = color 
        self.hover_start: float | None = None
        
        # New: hysteresis configuration (Problem 5 solution option B + A)
        self.hover_cooloff = 0.0 # Time when paddle left the button
        self.expansion_pad = 0

    def _is_hovered(self, positions) -> bool:
        # Check if any position is inside the *expanded* rect
        check_rect = self.rect.inflate(self.expansion_pad, self.expansion_pad)
        for pos in positions:
            if check_rect.collidepoint(pos[0], pos[1]):
                return True
        return False

    def update(self, positions) -> float:
        is_hovered = self._is_hovered(positions)
        now = time.time()
        
        if is_hovered:
            if self.hover_start is None:
                self.hover_start = now
            self.hover_cooloff = 0.0
            self.expansion_pad = 40  # Sticky box hysteresis
            return min((now - self.hover_start) / HOVER_DURATION, 1.0)
        else:
            # Not hovered -> grace period
            if self.hover_start is not None:
                if self.hover_cooloff == 0.0:
                    self.hover_cooloff = now
                elif now - self.hover_cooloff > 0.5: # 0.5s grace period
                    self.hover_start = None
                    self.hover_cooloff = 0.0
                    self.expansion_pad = 0
            else:
                self.expansion_pad = 0
                
        if self.hover_start is not None and self.hover_cooloff > 0.0:
            # During cooloff, freeze progress
            return min((self.hover_cooloff - self.hover_start) / HOVER_DURATION, 1.0)
            
        return 0.0

    def draw(self, surface: pygame.Surface, progress: float = 0.0):
        rgb_col = to_rgb(self.color)[:3]
        
        # Pulse alpha and add glow when hovered
        alpha = 100
        if progress > 0.0:
            # Pulsating opacity
            alpha = int(140 + 40 * math.sin(time.time() * 10))
            
            # Draw glowing halo
            halo_surf = pygame.Surface((self.rect.width + 20, self.rect.height + 20), pygame.SRCALPHA)
            pygame.draw.rect(halo_surf, (*rgb_col, 50), halo_surf.get_rect(), border_radius=12)
            surface.blit(halo_surf, (self.rect.x - 10, self.rect.y - 10))
        
        # Background rounded rect
        bg_surf = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        pygame.draw.rect(bg_surf, (*rgb_col, alpha), bg_surf.get_rect(), border_radius=8)
        surface.blit(bg_surf, self.rect)
        
        # 3D Highlight / Bevel
        r, g, b = rgb_col
        light = (min(r+100, 255), min(g+100, 255), min(b+100, 255))
        dark = (max(r-80, 0), max(g-80, 0), max(b-80, 0))
        
        # Dark bottom/right edge
        pygame.draw.rect(surface, dark, self.rect, 2, border_radius=8)
        
        # Light top/left edge overlay
        light_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width - 2, self.rect.height - 2)
        pygame.draw.rect(surface, light, light_rect, 1, border_radius=8)
        
        # Text centering
        text_surf = FONT_MEDIUM.render(self.label, True, (255, 255, 255))
        tx = self.rect.x + (self.rect.width - text_surf.get_width()) // 2
        ty = self.rect.y + (self.rect.height - text_surf.get_height()) // 2
        surface.blit(text_surf, (tx, ty))
        
        # Sleek progress bar at bottom
        if progress > 0:
            bar_w = int((self.rect.width - 16) * progress)
            bar_rect = pygame.Rect(self.rect.x + 8, self.rect.bottom - 10, bar_w, 4)
            pygame.draw.rect(surface, light, bar_rect, border_radius=2)

# ---------------------------------------------------------------------------
# UI manager
# ---------------------------------------------------------------------------
class UIManager:
    """Owns every menu screen, the HUD, and the pause-gesture detector."""

    def __init__(self):
        btn_w, btn_h = 180, 50
        cx = WIDTH // 2 - btn_w // 2

        # Home screen
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
            WIDTH // 2 - 90, 230, 180, 35, "CHANGE DIFFICULTY", (0, 140, 180),
        )

        self.selected_difficulty: int = 0
        self._difficulty_visible: bool = True

        # Pause screen
        px = WIDTH // 2 - 90
        self.pause_buttons: dict[str, HoverButton] = {
            "resume":  HoverButton(px, 160, 180, 50, "RESUME",  (0, 200, 0)),
            "restart": HoverButton(px, 230, 180, 50, "RESTART", (0, 200, 200)),
            "home":    HoverButton(px, 300, 180, 50, "HOME",    (255, 100, 0)),
        }

        self.result_buttons: dict[str, HoverButton] = {
            "replay":     HoverButton(cx, 300, btn_w, btn_h, "WATCH REPLAY", (0, 100, 200)),
            "play_again": HoverButton(cx, 370, btn_w, btn_h, "PLAY AGAIN",   (0, 200, 0)),
            "home":       HoverButton(cx, 440, btn_w, btn_h, "HOME MENU",    (255, 100, 0)),
        }

        self.p1_color: tuple = COLOR_P1
        self.p2_color: tuple = COLOR_P2

    def _draw_crt_overlay(self, surface: pygame.Surface):
        # Scanlines removed
        pass

    @staticmethod
    def _draw_glowing_cursor(surface: pygame.Surface, pos: tuple, color: tuple, radius: int = 12):
        pos_int = (int(pos[0]), int(pos[1]))
        rgb_col = to_rgb(color)[:3]
        
        # Pulsating outer glow
        pulse = 1.2 + 0.3 * math.sin(time.time() * 8)
        glow_rad = int(radius * pulse)
        glow_surf = pygame.Surface((glow_rad*2, glow_rad*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*rgb_col, 80), (glow_rad, glow_rad), glow_rad)
        surface.blit(glow_surf, (pos_int[0] - glow_rad, pos_int[1] - glow_rad))
        
        # Inner core white, outer ring color
        pygame.draw.circle(surface, (255, 255, 255), pos_int, radius - 4)
        pygame.draw.circle(surface, rgb_col, pos_int, radius, 3)

    @staticmethod
    def _outlined_text(surface: pygame.Surface, text: str, pos: tuple, font: pygame.font.Font, color: tuple,
                       outline_color=COLOR_BLACK, outline_thickness: int = 2):
        col_rgb = to_rgb(color)
        out_rgb = to_rgb(outline_color)
        # Fix: only support RGBA format explicitly when needed; text drawing can't naturally take RGBA length 4 tuples
        col_rgb = col_rgb[:3]
        out_rgb = out_rgb[:3]
        text_surf = font.render(text, True, col_rgb)
        out_surf = font.render(text, True, out_rgb)
        
        for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1),(0,-1),(0,1),(-1,0),(1,0)]:
            surface.blit(out_surf, (pos[0] + dx * outline_thickness, pos[1] + dy * outline_thickness))
        surface.blit(text_surf, pos)

    @staticmethod
    def _outline_glow_text(surface: pygame.Surface, text: str, y: int, font: pygame.font.Font, glow_color: tuple, glow_size: int = 4):
        glow_rgb = to_rgb(glow_color)[:3]
        text_surf = font.render(text, True, glow_rgb)
        x = (WIDTH - text_surf.get_width()) // 2
        
        # Draw multiple times with lower alpha for glow effect
        # Pygame surface alpha works best with an intermediate surface
        glow_surf = pygame.Surface((text_surf.get_width() + glow_size*4, text_surf.get_height() + glow_size*4), pygame.SRCALPHA)
        for offset in range(glow_size, 0, -1):
            alpha = int(100 / glow_size)
            temp = font.render(text, True, (*glow_rgb, alpha))
            
            for dx, dy in [(-offset,-offset),(-offset,offset),(offset,-offset),(offset,offset),(0,-offset),(0,offset),(-offset,0),(offset,0)]:
                glow_surf.blit(temp, (glow_size*2 + dx, glow_size*2 + dy))
        
        surface.blit(glow_surf, (x - glow_size*2, y - glow_size*2))

    @staticmethod
    def _draw_centered_text(surface: pygame.Surface, text: str, y: int, font: pygame.font.Font, color: tuple,
                            outline_color=COLOR_BLACK, outline_thickness: int = 2):
        text_surf = font.render(text, True, to_rgb(color)[:3])
        x = (WIDTH - text_surf.get_width()) // 2
        UIManager._outlined_text(surface, text, (x, y), font, color, outline_color, outline_thickness)

    def draw_click_calibration(self, surface: pygame.Surface, step: int) -> None:
        over = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        over.fill((0, 0, 0, 140))
        surface.blit(over, (0, 0))

        self._draw_centered_text(surface, "CALIBRATION", 60, FONT_TITLE, COLOR_WHITE)
        
        if step == 0:
            self._draw_centered_text(surface, "Click anywhere on the CYAN paddle", 200, FONT_LARGE, self.p1_color)
            UIManager._draw_glowing_cursor(surface, pygame.mouse.get_pos(), self.p1_color, 10)
        elif step == 1:
            self._draw_centered_text(surface, "Click anywhere on the PINK paddle", 200, FONT_LARGE, self.p2_color)
            UIManager._draw_glowing_cursor(surface, pygame.mouse.get_pos(), self.p2_color, 10)

    def draw_home(self, surface: pygame.Surface, p1_pos, p2_pos) -> str | None:
        over = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        over.fill((0, 0, 0, 150))
        surface.blit(over, (0, 0))
        
        # Retro Glow Title
        t = time.time()
        base_color = (0, 150, 255) # Vibrant retro electric blue
        c1 = (int(127 + 127*math.sin(t*3)), 255, 255)
        c2 = (255, int(127 + 127*math.cos(t*2)), 255)
        self._outline_glow_text(surface, "COLOR WAR", 60, FONT_TITLE, c1, glow_size=4)
        self._draw_centered_text(surface, "COLOR WAR", 60, FONT_TITLE, base_color, c2, 1)
        
        self._draw_centered_text(surface, "Hover paddle over button to select", 130, FONT_SMALL, COLOR_GRAY)
        self._draw_centered_text(surface, f"Difficulty:{DIFFICULTY_NAMES[self.selected_difficulty]}", 260, FONT_MEDIUM, COLOR_WHITE)

        positions = [p1_pos, p2_pos]
        action = None

        for key, btn in self.home_buttons.items():
            if key == "toggle_diff" and self._difficulty_visible:
                continue
            prog = btn.update(positions)
            btn.draw(surface, prog)
            if prog >= 1.0:
                btn.hover_start = None
                if key == "start":
                    action = "start"
                elif key == "toggle_diff":
                    self._difficulty_visible = True

        if self._difficulty_visible:
            for key, btn in self.diff_buttons.items():
                prog = btn.update(positions)
                btn.draw(surface, prog)
                if prog >= 1.0:
                    btn.hover_start = None
                    self.selected_difficulty = int(key.split("_")[1])
                    self._difficulty_visible = False
        UIManager._draw_glowing_cursor(surface, p1_pos, self.p1_color)
        UIManager._draw_glowing_cursor(surface, p2_pos, self.p2_color)

        self._draw_centered_text(surface, "Hold both paddles at TOP to PAUSE during game", HEIGHT - 30, FONT_SMALL, COLOR_GRAY)
        self._draw_crt_overlay(surface)
        return action

    def draw_pause(self, surface: pygame.Surface, p1_pos, p2_pos) -> str | None:
        over = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        over.fill((0, 0, 0, 160))
        surface.blit(over, (0, 0))
        
        self._draw_centered_text(surface, "PAUSED", 80, FONT_TITLE, COLOR_WHITE)

        positions = [p1_pos, p2_pos]
        action = None
        for key, btn in self.pause_buttons.items():
            prog = btn.update(positions)
            btn.draw(surface, prog)
            if prog >= 1.0:
                btn.hover_start = None
                action = key
        UIManager._draw_glowing_cursor(surface, p1_pos, self.p1_color)
        UIManager._draw_glowing_cursor(surface, p2_pos, self.p2_color)
        
        self._draw_crt_overlay(surface)
        return action

    def draw_results(self, surface: pygame.Surface, winner_text: str, p1_pct: int, p2_pct: int, p1_pos, p2_pos) -> str | None:
        over = pygame.Surface((WIDTH - 160, HEIGHT - 80), pygame.SRCALPHA)
        over.fill((0, 0, 0, 200))
        surface.blit(over, (80, 40))
        pygame.draw.rect(surface, to_rgb(COLOR_WHITE)[:3], (80, 40, WIDTH - 160, HEIGHT - 80), 3)

        self._draw_centered_text(surface, winner_text, 70, FONT_TITLE, COLOR_WHITE)

        bar_w = 300
        self._outlined_text(surface, f"P1: {p1_pct}%", (120, 150), FONT_LARGE, self.p1_color)
        pygame.draw.rect(surface, to_rgb(self.p1_color)[:3], (120, 200, int(p1_pct / 100 * bar_w), 25))
        
        self._outlined_text(surface, f"P2: {p2_pct}%", (120, 230), FONT_LARGE, self.p2_color)
        pygame.draw.rect(surface, to_rgb(self.p2_color)[:3], (120, 280, int(p2_pct / 100 * bar_w), 25))

        positions = [p1_pos, p2_pos]
        action = None
        for key, btn in self.result_buttons.items():
            prog = btn.update(positions)
            btn.draw(surface, prog)
            if prog >= 1.0:
                btn.hover_start = None
                action = key
        UIManager._draw_glowing_cursor(surface, p1_pos, self.p1_color)
        UIManager._draw_glowing_cursor(surface, p2_pos, self.p2_color)
        
        self._draw_crt_overlay(surface)
        return action

    def draw_hud(self, surface: pygame.Surface, time_left: int, difficulty: int, level: int = 1, level_up_until: float = 0,
                 combo_count: int = 0, combo_owner: str | None = None, combo_banner_until: float = 0,
                 p1_color: tuple = COLOR_P1, p2_color: tuple = COLOR_P2):
        
        # Pause target region
        m = 40
        pygame.draw.rect(surface, (60, 60, 60), (m, 2, WIDTH - 2*m, PAUSE_TOP_THRESHOLD - 2), 2)
        self._outlined_text(surface, "PAUSE ZONE", (WIDTH // 2 - 40, PAUSE_TOP_THRESHOLD - 20), FONT_SMALL, COLOR_GRAY)

        now = time.time()
        
        if time_left <= ENDING_WARNING_SECONDS and time_left > 0:
            pulse = abs(np.sin(now * 6))
            timer_bg = (int(180 * pulse), 0, 0)
            timer_border = (255, 0, 0)
            timer_text_col = (255, 0, 0)
        else:
            timer_bg = to_rgb(COLOR_BLACK)[:3]
            timer_border = to_rgb(COLOR_WHITE)[:3]
            timer_text_col = to_rgb(COLOR_WHITE)[:3]

        pygame.draw.rect(surface, timer_bg, (WIDTH // 2 - 40, 5, 80, 40))
        pygame.draw.rect(surface, timer_border, (WIDTH // 2 - 40, 5, 80, 40), 2)
        text_surf = FONT_LARGE.render(str(time_left), True, timer_text_col)
        surface.blit(text_surf, (WIDTH // 2 - text_surf.get_width()//2, 25 - text_surf.get_height()//2))

        self._outlined_text(surface, DIFFICULTY_NAMES[difficulty], (10, 10), FONT_MEDIUM, COLOR_GRAY)
        self._outlined_text(surface, f"LVL {level}", (WIDTH - 80, 10), FONT_MEDIUM, COLOR_WHITE)

        # Combo
        if combo_count >= COMBO_MIN_HITS and now < combo_banner_until:
            rem = combo_banner_until - now
            fade = min(1.0, rem / (COMBO_BANNER_DURATION * 0.5))
            b_col = p1_color if combo_owner == "P1" else p2_color
            b_rgb = to_rgb(b_col)[:3]
            b_rgb_fade = tuple(int(c * fade) for c in b_rgb)
            self._draw_centered_text(surface, f"COMBO x{combo_count}!", HEIGHT - 60, FONT_LARGE, b_rgb_fade, (0,0,0), 3)

        # Level up
        if now < level_up_until:
            from .config import LEVEL_OVERLAY_DURATION
            fade = min(1.0, (level_up_until - now) / (LEVEL_OVERLAY_DURATION * 0.6))
            alpha_col = tuple(int(c * fade) for c in (255, 255, 0))
            self._draw_centered_text(surface, f"LEVEL {level}!", HEIGHT // 2 - 30, FONT_TITLE, alpha_col)

    def draw_pause_progress(self, surface: pygame.Surface):
        if self.pause_hold_start is not None:
            prog = (time.time() - self.pause_hold_start) / PAUSE_HOLD_DURATION
            self._draw_centered_text(surface, f"Pausing... {int(prog * 100)}%", HEIGHT - 40, FONT_MEDIUM, COLOR_WHITE)

    def draw_prestart(self, surface: pygame.Surface, prestart_until: float):
        remaining = max(0, prestart_until - time.time())
        over = pygame.Surface((300, 100), pygame.SRCALPHA)
        over.fill((0, 0, 0, 180))
        surface.blit(over, (WIDTH // 2 - 150, HEIGHT // 2 - 50))
        self._draw_centered_text(surface, f"GET READY {int(remaining) + 1}", HEIGHT // 2 - 30, FONT_LARGE, COLOR_WHITE)
        self._draw_centered_text(surface, "Stay on your side!", HEIGHT // 2 + 10, FONT_MEDIUM, (255, 200, 0))

    def draw_hints(self, surface: pygame.Surface, help_show_until: float):
        fade = min(1.0, (help_show_until - time.time()) / 2.0)
        over = pygame.Surface((340, 140), pygame.SRCALPHA)
        over.fill((0, 0, 0, int(200 * fade)))
        surface.blit(over, (WIDTH // 2 - 170, HEIGHT // 2 - 70))
        
        alpha_col = tuple(int(c * fade) for c in to_rgb(COLOR_WHITE)[:3])
        hints = [
            "Hit the ball to paint the arena!",
            "Hold BOTH paddles at TOP to pause",
            "Most paint coverage wins",
            "Q = Quit",
        ]
        for i, txt in enumerate(hints):
            self._draw_centered_text(surface, txt, HEIGHT // 2 - 50 + i * 28, FONT_MEDIUM, alpha_col, (0,0,0), 1)

    def draw_replay_hud(self, surface: pygame.Surface, idx: int, total: int):
        from .config import REPLAY_DURATION
        progress = (idx + 1) / total
        bar_h, bar_y = 8, HEIGHT - 12
        pygame.draw.rect(surface, to_rgb(COLOR_GRAY)[:3], (10, bar_y, WIDTH - 20, bar_h))
        pygame.draw.rect(surface, to_rgb(COLOR_WHITE)[:3], (10, bar_y, int((WIDTH - 20) * progress), bar_h))
        self._outlined_text(surface, "REPLAY", (15, 10), FONT_LARGE, (255, 200, 0))
        speed_label = f"{total / (REPLAY_DURATION * 30):.1f}x"
        self._outlined_text(surface, speed_label, (WIDTH - 80, 20), FONT_MEDIUM, COLOR_WHITE)

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

    def reset_hover_states(self):
        for btn in self.home_buttons.values(): btn.hover_start = None
        for btn in self.diff_buttons.values(): btn.hover_start = None
        for btn in self.pause_buttons.values(): btn.hover_start = None
        for btn in self.result_buttons.values(): btn.hover_start = None
        self.pause_hold_start = None
