"""Generate poster images AND show live contour overlay for screen-recording.

Usage (from the project root, one level above ColorWar/):
    python -m ColorWar.generate_poster_images

Controls:
    S  – Save poster grid images to poster_images/
    Q  – Quit
"""

import cv2
import numpy as np
import os
import sys

# ------------------------------------------------------------------
# Imports from the game package
# ------------------------------------------------------------------
from .config import (
    PADDLE_HSV_PRIORS, HSV_BLUR_KSIZE, WIDTH, HEIGHT,
    CALIB_LOCK_FRAMES,
)
from .tracking import PaddleTracker

os.makedirs("poster_images", exist_ok=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def add_label(img, text, color=(255, 255, 255)):
    """Draw a labelled heading on an image (with dark shadow)."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
    cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    return img


def create_multicue_images(frame, tracker: PaddleTracker):
    """Return individual images for each multi-cue detection stage."""
    raw = frame.copy()

    # 1. HSV colour mask (pink paddle)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_blurred = cv2.GaussianBlur(hsv, (HSV_BLUR_KSIZE, HSV_BLUR_KSIZE), 0)
    color_mask = tracker._get_mask(hsv_blurred, False)  # pink
    color_overlay = np.zeros_like(frame)
    color_overlay[color_mask > 0] = (200, 100, 255)

    # 2. Canny edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_vis[edges > 0] = (255, 255, 0)

    return [
        ("1_raw_rgb_frame.jpg", raw),
        ("2_hsv_colour_mask.jpg", color_overlay),
        ("3_canny_edge_map.jpg", edges_vis),
    ]


def create_watershed_images(frame, tracker: PaddleTracker):
    """Return individual images for each watershed extraction stage."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_blurred = cv2.GaussianBlur(hsv, (HSV_BLUR_KSIZE, HSV_BLUR_KSIZE), 0)
    mask = tracker._get_mask(hsv_blurred, False)  # pink

    # Distance transform
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)
    heatmap[mask == 0] = 0

    # Watershed markers
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_bg = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=3)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    marker_vis = np.zeros_like(frame)
    marker_vis[markers == 1] = (50, 0, 0)
    marker_vis[markers > 1] = (0, 255, 0)
    marker_vis[unknown == 255] = (0, 0, 255)

    # Final: actual tracked contour
    final_output = frame.copy()
    actual_contour = tracker._state[False]._last_contour
    if actual_contour is not None:
        cv2.drawContours(final_output, [actual_contour], -1, (200, 100, 255), 4)

    # Convert mask to BGR for saving
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return [
        ("4_raw_colour_mask.jpg", mask_bgr),
        ("5_distance_transform.jpg", heatmap),
        ("6_watershed_seeds.jpg", marker_vis),
        ("7_exact_contour_extraction.jpg", final_output),
    ]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    tracker = PaddleTracker()
    tracker.reset_to_priors()
    tracker.in_game = True  # use game-mode rules (never drop contour)

    # Skip cyan — only detect pink
    tracker._state[True].locked = True

    print("\n" + "=" * 55)
    print("🎨  Color War – Poster & Live Contour Viewer  🎨")
    print("=" * 55)
    print("  • Wave the PINK paddle to calibrate")
    print("  • Live contour drawn on the feed for screen recording")
    print("  • Press  S  to save poster grid images")
    print("  • Press  Q  to quit")
    print("=" * 55 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # --- Exact game pipeline ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_blurred = tracker.blur_hsv(hsv)
        tracker.begin_frame(frame)

        st_pink = tracker._state[False]
        if not st_pink.locked:
            tracker.calibrate_step(hsv, hsv_blurred)
        else:
            pos, contour = tracker.track_paddle(hsv_blurred, False)

        tracker.end_frame()

        # --- Draw live contour (pink only) ---
        display = frame.copy()

        contour = st_pink._last_contour
        if contour is not None:
            cv2.drawContours(display, [contour], -1, (200, 100, 255), 3)
            bx, by, bw, bh = cv2.boundingRect(contour)
            cv2.putText(display, "PINK", (bx, by - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 255), 2)

        # Status overlay
        if not st_pink.locked:
            pct = int(st_pink.lock_counter / CALIB_LOCK_FRAMES * 100)
            cv2.putText(display, f"Wave pink paddle...  {pct}%", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(display, "PINK OK  |  S=save  Q=quit",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Color War - Live Contours", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and st_pink.locked:
            print("Capturing frame and generating images...")
            all_images = create_multicue_images(frame, tracker) + create_watershed_images(frame, tracker)
            for fname, img in all_images:
                path = os.path.join("poster_images", fname)
                cv2.imwrite(path, img)
                print(f"  ✅ {path}")

        elif key == ord('q'):
            print("Done.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
