import cv2
import numpy as np
import sys

# ============================================================
# Canny (live) + Multiscale Template Matching on webcam feed
# ============================================================

DEFAULT_TEMPLATE_PATH = r"rasp\Screenshot 2026-02-09 210243.png"
DEFAULT_CAMERA_INDEX  = 0

INIT_LOW        = 50
INIT_RATIO_X10  = 30
INIT_MATCH_THR  = 70

# ── multiscale parameters ───────────────────────────────────
SCALE_MIN   = 0.9    # smallest scale to try (template shrinks relative to frame)
SCALE_MAX   = 3.0    # largest scale to try
SCALE_STEP  = 0.6   # increment between scales (smaller = finer, slower)

# ── globals ─────────────────────────────────────────────────
template_gray   = None
template_color  = None
img_blur        = None
edges           = None

MATCH_METHOD = cv2.TM_CCOEFF_NORMED


# ────────────────────────────────────────────────────────────
def canny_callback(_=0):
    global edges, img_blur
    if img_blur is None:
        return
    low   = max(0,  cv2.getTrackbarPos('Low Threshold', 'Trackbars'))
    ratio = max(20, cv2.getTrackbarPos('Ratio ×10',     'Trackbars')) / 10.0
    edges = cv2.Canny(img_blur, low, int(low * ratio))
    cv2.imshow('Canny Edges', edges)


def clamp_ratio():
    val = cv2.getTrackbarPos('Ratio ×10', 'Trackbars')
    if val < 20:
        cv2.setTrackbarPos('Ratio ×10', 'Trackbars', 20)
        canny_callback()


# ────────────────────────────────────────────────────────────
def do_multiscale_match(frame_gray, frame_color):
    """
    Searches for template_gray in frame_gray across a range of frame scales.

    Strategy:
      - Keep template fixed.
      - Resize the frame at each scale in [SCALE_MIN, SCALE_MAX].
      - Run matchTemplate at each scale.
      - Track the scale and location that produced the highest score.
      - Map the winning location back to original-frame coordinates.

    Returns: annotated BGR frame, best confidence (float).
    """
    if template_gray is None:
        return frame_color.copy(), 0.0

    th, tw     = template_gray.shape[:2]
    fh, fw     = frame_gray.shape[:2]
    match_thr  = cv2.getTrackbarPos('Match Thr ×100', 'Trackbars') / 100.0

    best_val   = -np.inf
    best_loc   = None       # (x, y) in the SCALED frame
    best_scale = 1.0        # scale factor that produced best_val

    # ── pyramid search ──────────────────────────────────────
    scale = SCALE_MIN
    while scale <= SCALE_MAX + 1e-6:   # +epsilon avoids float precision cutoff

        new_w = int(fw * scale)
        new_h = int(fh * scale)

        # Template must fit inside the scaled frame
        if new_w < tw or new_h < th:
            scale += SCALE_STEP
            continue

        # Resize the frame (NOT the template)
        resized = cv2.resize(frame_gray, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

        result = cv2.matchTemplate(resized, template_gray, MATCH_METHOD)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val   = max_val
            best_loc   = max_loc
            best_scale = scale

        scale += SCALE_STEP

    # ── map best_loc back to original frame coordinates ─────
    annotated = frame_color.copy()

    if best_loc is None or best_val < match_thr:
        label = f"No match (best={best_val:.2f})"
        cv2.putText(annotated, label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return annotated, best_val

    # Coordinates in the scaled frame → divide by scale → original frame coords
    x_orig = int(best_loc[0] / best_scale)
    y_orig = int(best_loc[1] / best_scale)

    # Bounding box size in original frame = template size / scale
    # (the template represents a region that, at best_scale, was tw×th pixels)
    box_w = int(tw / best_scale)
    box_h = int(th / best_scale)

    top_left     = (x_orig, y_orig)
    bottom_right = (x_orig + box_w, y_orig + box_h)

    cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 2)

    label = f"Match: {best_val:.2f}  scale: {best_scale:.2f}"
    cv2.putText(annotated, label,
                (top_left[0], max(top_left[1] - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return annotated, best_val


# ────────────────────────────────────────────────────────────
def main():
    global template_gray, template_color, img_blur, edges

    template_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEMPLATE_PATH
    cam_index     = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CAMERA_INDEX

    # ── load template ────────────────────────────────────────
    template_color = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template_color is None:
        print(f"[ERROR] Cannot load template: {template_path}")
        sys.exit(1)

    template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)

    print(f"[INFO] Template size: {template_gray.shape[1]}w × {template_gray.shape[0]}h px")
    print(f"[INFO] Scale search: {SCALE_MIN} → {SCALE_MAX}, step {SCALE_STEP}")
    print(f"[INFO] Total scales to evaluate per frame: "
          f"{int((SCALE_MAX - SCALE_MIN) / SCALE_STEP) + 1}")

    # ── open camera ──────────────────────────────────────────
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {cam_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── windows ──────────────────────────────────────────────
    for name in ('Live Feed', 'Canny Edges', 'Template', 'Trackbars'):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow('Template', template_color)

    # ── trackbars ────────────────────────────────────────────
    cv2.createTrackbar('Low Threshold',  'Trackbars', INIT_LOW,       200, canny_callback)
    cv2.createTrackbar('Ratio ×10',      'Trackbars', INIT_RATIO_X10,  40, canny_callback)
    cv2.createTrackbar('Match Thr ×100', 'Trackbars', INIT_MATCH_THR, 100, lambda _: None)
    try:
        cv2.setTrackbarMin('Ratio ×10', 'Trackbars', 20)
    except Exception:
        pass
    clamp_ratio()

    print("[INFO] Press 'q' or ESC to quit.")

    # ── main loop ────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Empty frame, retrying...")
            continue

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_blur   = cv2.GaussianBlur(frame_gray, (5, 5), 1.4)

        low   = max(0,  cv2.getTrackbarPos('Low Threshold', 'Trackbars'))
        ratio = max(20, cv2.getTrackbarPos('Ratio ×10',     'Trackbars')) / 10.0
        edges = cv2.Canny(img_blur, low, int(low * ratio))

        annotated, confidence = do_multiscale_match(frame_gray, frame)

        cv2.imshow('Live Feed',   annotated)
        cv2.imshow('Canny Edges', edges)

        clamp_ratio()
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()