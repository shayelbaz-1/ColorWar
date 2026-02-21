import cv2
import numpy as np
import json
import time
import os
from config import *

# Ensure outputs directory exists
os.makedirs("poster_images", exist_ok=True)


def add_label(img, text, color=(255,255,255)):
    # Standardize image to BGR so we can draw colored text and stack them
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 6)
    cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    return img

def create_multicue_grid(frame, p1_calib, p2_calib, bg_subtractor):
    raw = frame.copy()
    
    fg_mask = bg_subtractor.apply(frame)
    fg_mask_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_blurred = cv2.GaussianBlur(hsv, (HSV_BLUR_KSIZE, HSV_BLUR_KSIZE), 0)
    
    lower = np.array([p2_calib["h_min"], p2_calib["s_min"], p2_calib["v_min"]])
    upper = np.array([p2_calib["h_max"], p2_calib["s_max"], p2_calib["v_max"]])
    color_mask = cv2.inRange(hsv_blurred, lower, upper)
    
    color_overlay = np.zeros_like(frame)
    color_overlay[color_mask > 0] = (200, 100, 255) # Pinkish
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[edges > 0] = (255, 255, 0) # Cyan in BGR
    
    raw = add_label(raw, "1. Raw RGB Frame")
    fg_mask_colored = add_label(fg_mask_colored, "2. Motion Mask (MOG2)")
    color_overlay = add_label(color_overlay, "3. HSV Adaptive Threshold")
    edges_colored = add_label(edges_colored, "4. Canny Edge Map")
    
    top = np.hstack([raw, fg_mask_colored])
    bottom = np.hstack([color_overlay, edges_colored])
    return np.vstack([top, bottom])

def create_watershed_sequence(frame, p1_calib, p2_calib):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_blurred = cv2.GaussianBlur(hsv, (HSV_BLUR_KSIZE, HSV_BLUR_KSIZE), 0)
    
    lower = np.array([p2_calib["h_min"], p2_calib["s_min"], p2_calib["v_min"]])
    upper = np.array([p2_calib["h_max"], p2_calib["s_max"], p2_calib["v_max"]])
    mask = cv2.inRange(hsv_blurred, lower, upper)
    
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)
    # Zero out background in heatmap so it clearly shows distance from edge
    heatmap[mask == 0] = 0
    
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_bg = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=3)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    marker_vis = np.zeros_like(frame)
    marker_vis[markers == 1] = (50, 0, 0) # Background -> dark blueish
    marker_vis[markers > 1] = (0, 255, 0) # Foreground seeds -> green
    marker_vis[unknown == 255] = (0, 0, 255) # Unknown boundary -> red
    
    markers_ws = cv2.watershed(frame.copy(), markers)
    final_output = frame.copy()
    
    boundary_mask = np.zeros_like(mask)
    boundary_mask[markers_ws == -1] = 255
    contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        best_c = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_output, [best_c], -1, (200, 100, 255), 4)
        
    mask_bgr = add_label(mask.copy(), "1. Raw Color Mask")
    heatmap = add_label(heatmap, "2. Distance Transform")
    marker_vis = add_label(marker_vis, "3. Watershed Seeds", color=(100,255,100))
    final_output = add_label(final_output, "4. Exact Contour Extraction")

    top = np.hstack([mask_bgr, heatmap])
    bottom = np.hstack([marker_vis, final_output])
    return np.vstack([top, bottom])

def main():
    # Use the default hardware-tuned priors from config.py instead of a saved JSON
    p1 = PADDLE_HSV_PRIORS["CYAN"]
    p2 = PADDLE_HSV_PRIORS["PINK"]
    # Using camera 0 automatically
    cap = cv2.VideoCapture(0)
    # Try high res for nice posters
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    print("\n" + "="*50)
    print("🎨 Color War Poster Image Generator 🎨")
    print("="*50)
    print("1. Stand back with paddles in view.")
    print("2. Move around a bit so the motion mask kicks in.")
    print("3. Press 'S' to save the poster grid images!")
    print("4. Press 'Q' to quit without saving.")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        bg_subtractor.apply(frame)

        preview = frame.copy()
        cv2.putText(preview, "Press 'S' to save poster images! (Q to quit)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Poster Generator Preview", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("Capturing frame and generating visualizations...")
            
            grid1 = create_multicue_grid(frame, p1, p2, bg_subtractor)
            grid2 = create_watershed_sequence(frame, p1, p2)
            
            p1 = "poster_images/multicue_feature_grid.jpg"
            p2 = "poster_images/watershed_extraction_grid.jpg"
            cv2.imwrite(p1, grid1)
            cv2.imwrite(p2, grid2)
            
            print(f"✅ Saved `{p1}`!")
            print(f"✅ Saved `{p2}`!")
            break
            
        elif key == ord('q'):
            print("Cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
