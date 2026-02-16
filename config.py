"""Central configuration for Color War Pong."""

# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 640, 480

# ---------------------------------------------------------------------------
# Ball
# ---------------------------------------------------------------------------
BALL_RADIUS = 15
BALL_INITIAL_SPEED = 6.0
BALL_MIN_SPEED = 8.0
BALL_MAX_SPEED = 25.0
BALL_FRICTION = 0.995

# ---------------------------------------------------------------------------
# Paddle
# ---------------------------------------------------------------------------
PADDLE_RADIUS = 30
MIN_CONTOUR_AREA = 400
TRACKING_MAX_JUMP = 80          # max px jump per frame before suspicion
TRACKING_MISS_FRAMES = 8        # consecutive misses before fallback
TRACKING_REACQUIRE_SECONDS = 1.5
TRACKING_REACQUIRE_MAX_JUMP = 200
TRACKING_ASPECT_RATIO_MAX = 3.5
PADDLE_BLOB_COLLISION_PADDING = 5

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
GAME_DURATION = 60          # seconds
PRESTART_DURATION = 5       # seconds of idle before midline enforcement
POWERUP_SPAWN_TIME = 15     # seconds (rarer powerups)
MEGA_BRUSH_DURATION = 5     # seconds
HOVER_DURATION = 2.0        # seconds to trigger hover-selection
PAUSE_HOLD_DURATION = 2.0   # seconds holding paddles at top to pause
PAUSE_TOP_THRESHOLD = 60    # y-pixels from top edge

# ---------------------------------------------------------------------------
# Fixed player colors (BGR) — cyan for P1, pink for P2
# ---------------------------------------------------------------------------
COLOR_P1 = (255, 255, 0)        # Cyan  (BGR)
COLOR_P2 = (180, 0, 255)        # Pink  (BGR)
COLOR_POWERUP = (0, 255, 0)     # Green
COLOR_BOMB = (0, 0, 200)        # Dark red
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (100, 100, 100)
COLOR_MIDLINE = (200, 200, 200)

# ---------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------
PARTICLE_COLORS = [
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (255, 255, 255),
]

# ---------------------------------------------------------------------------
# Game states
# ---------------------------------------------------------------------------
STATE_AUTODETECT = "AUTODETECT"
STATE_HOME = "HOME"
STATE_GAME = "GAME"
STATE_PAUSE = "PAUSE"
STATE_RESULTS = "RESULTS"
STATE_REPLAY = "REPLAY"

# ---------------------------------------------------------------------------
# Difficulty
# ---------------------------------------------------------------------------
DIFF_EASY = 0
DIFF_MEDIUM = 1
DIFF_HARD = 2
DIFF_INSANE = 3

DIFFICULTY_NAMES = ["EASY", "MEDIUM", "HARD", "INSANE"]

DIFFICULTY_CONFIG = {
    DIFF_EASY: {
        "speed_mult": 1.0,
        "multi_ball": False,
        "fog": False,
        "transparency": False,
        "obstacles": False,
        "bomb": False,
    },
    DIFF_MEDIUM: {
        "speed_mult": 1.3,
        "multi_ball": True,
        "fog": False,
        "transparency": False,
        "obstacles": False,
        "bomb": False,
    },
    DIFF_HARD: {
        "speed_mult": 1.5,
        "multi_ball": True,
        "fog": True,
        "transparency": False,
        "obstacles": True,
        "bomb": False,
    },
    DIFF_INSANE: {
        "speed_mult": 1.8,
        "multi_ball": True,
        "fog": True,
        "transparency": True,
        "obstacles": True,
        "bomb": True,
    },
}

# ---------------------------------------------------------------------------
# Obstacles
# ---------------------------------------------------------------------------
OBSTACLE_COLOR = (180, 180, 180)
OBSTACLE_THICKNESS = 3
OBSTACLES_PER_SIDE = 2
OBSTACLE_REFRESH_SECONDS = 20

# ---------------------------------------------------------------------------
# Paddle HSV auto-calibration (pure colour-blob, no deep learning)
# ---------------------------------------------------------------------------
# Broad HSV priors for the two paddle colours  (H, S, V ranges in OpenCV 0-179)
# Cyan paddle: hue ~75-105, high saturation required
# Pink paddle: hue ~138-172 (wider to catch angled/thin views), sat 90+ to reject skin
PADDLE_HSV_PRIORS = {
    "CYAN": {"h_min": 75, "h_max": 105, "s_min": 100, "s_max": 255, "v_min": 80, "v_max": 255},
    "PINK": {"h_min": 138, "h_max": 172, "s_min": 90, "s_max": 255, "v_min": 70, "v_max": 255},
}

# Hard HSV floors — refinement / adaptation can NEVER go below these.
# These prevent drift into skin tones (low-sat, wide-hue) or brown.
HSV_FLOOR_CYAN  = {"s_min": 80,  "v_min": 60}
HSV_FLOOR_PINK  = {"s_min": 85, "v_min": 55, "h_min": 133, "h_max": 178}
# ^ pink s_min 85 (skin is <80), wider hue band to survive angled views

# HSV pre-processing
HSV_BLUR_KSIZE = 5               # Gaussian blur kernel for noise suppression

# HSV refinement tolerances (used during calibration sampling)
HSV_REFINE_HUE_TOL = 12          # slightly wider hue window for edge-on colour shifts
HSV_REFINE_SV_TOL = 35           # slightly wider S/V to cover angled lighting

# Adaptive HSV during gameplay
HSV_ADAPTIVE_INTERVAL = 30       # frames between adaptive updates
HSV_ADAPTIVE_BLEND = 0.2         # conservative blend to avoid fast drift

# Calibration: how many stable frames before a paddle is considered "locked"
CALIB_LOCK_FRAMES = 25           # ~0.8s at 30fps — slightly more stringent
CALIB_LOCK_DECAY = 1             # lock_counter decrease per missed frame (was 2)
PINK_CALIB_SAT_MIN = 90          # pink candidates must have avg_sat >= this (skin is ~60-80)

# Contour quality thresholds (shape-based false-positive rejection)
CONTOUR_SOLIDITY_MIN = 0.45      # area / convexHullArea — relaxed for edge-on views
CONTOUR_COMPACTNESS_MIN = 0.20   # 4*pi*area / perimeter^2 — relaxed for thin blobs
CONTOUR_AREA_MAX = 25000         # reject huge blobs (entire shirt detected)

# Runtime confidence — hysteresis gate (0.0 – 1.0)
TRACKING_CONFIDENCE_MIN = 0.25   # threshold to START showing real contour
CONFIDENCE_HOLD = 0.12           # lower threshold to KEEP contour (hysteresis)
CONFIDENCE_GRACE_FRAMES = 3      # frames of grace before dropping to circle

# ---------------------------------------------------------------------------
# Classical per-frame detection (color + edge + shape — no temporal trackers)
# ---------------------------------------------------------------------------
# Score fusion weights for per-frame candidate scoring (should sum to ~1.0)
# Rebalanced: color is most reliable, proximity helps continuity, shape reduced
# because it penalises edge-on views unfairly.
DETECT_COLOR_WEIGHT = 0.45       # HSV in-range purity + saturation (primary cue)
DETECT_EDGE_WEIGHT = 0.20        # Canny edge density inside contour boundary
DETECT_SHAPE_WEIGHT = 0.15       # solidity, compactness, area, aspect ratio (relaxed)
DETECT_PROXIMITY_WEIGHT = 0.20   # closeness to previous frame position (continuity)

# Canny edge detection thresholds
EDGE_CANNY_LOW = 50
EDGE_CANNY_HIGH = 150

# Position smoothing (exponential moving average)
EMA_ALPHA = 0.6                  # 0=full smoothing 1=no smoothing (responsive)

# Optional motion-based foreground mask (cv2 BackgroundSubtractorMOG2)
DETECT_USE_MOTION_MASK = True    # enable foreground mask to reject static distractors
MOTION_MASK_LEARNING_RATE = 0.005  # slow learning so static scene becomes background
MOTION_MASK_MIN_OVERLAP = 0.15   # min fraction of contour overlapping motion fg (relaxed)
MOTION_PENALTY_MULT = 0.5       # penalty multiplier when overlap is below threshold (was 0.3)

# Edge-on relaxed detection (thin paddle views)
MIN_CONTOUR_AREA_EDGEON = 30     # min px area for edge-on candidates (very thin)
EDGE_ON_ASPECT_MAX = 14.0        # max bounding-rect aspect ratio (very thin allowed)

# ---------------------------------------------------------------------------
# Level progression (in-game)
# ---------------------------------------------------------------------------
LEVEL_INTERVAL_SECONDS = 10
LEVEL_SPEED_MULT = 1.05
BALL_ADD_INTERVAL_SECONDS = 20
MAX_BALLS = 3
LEVEL_OVERLAY_DURATION = 1.5

# ---------------------------------------------------------------------------
# Audio feedback (procedural pygame tones)
# ---------------------------------------------------------------------------
PROGRESS_BEEP_INTERVAL = 5
PROGRESS_BEEP_MIN_FREQ = 400
PROGRESS_BEEP_MAX_FREQ = 900
ENDING_WARNING_SECONDS = 10
ENDING_JINGLE_INTERVAL = 1.0

# ---------------------------------------------------------------------------
# Screen shake
# ---------------------------------------------------------------------------
SHAKE_DURATION = 0.15
SHAKE_INTENSITY_HIT = 4
SHAKE_INTENSITY_LEVEL = 8

# ---------------------------------------------------------------------------
# Combo system
# ---------------------------------------------------------------------------
COMBO_MIN_HITS = 3
COMBO_TIMEOUT = 4.0
COMBO_BANNER_DURATION = 1.5

# ---------------------------------------------------------------------------
# Confetti
# ---------------------------------------------------------------------------
CONFETTI_COUNT_LEVEL = 60
CONFETTI_COUNT_COMBO = 40
CONFETTI_COLORS = [
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 0),
    (255, 100, 100),
    (100, 100, 255),
    (255, 255, 255),
]

# ---------------------------------------------------------------------------
# Timelapse replay
# ---------------------------------------------------------------------------
REPLAY_SAMPLE_EVERY = 3
REPLAY_DURATION = 6.0
