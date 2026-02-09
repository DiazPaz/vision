import cv2 as cv
import numpy as np
import time
from collections import deque

TEMPLATE_PATH = "template.png"
CAM_INDEX = 0

FRAME_MAX_W = 640
ROI_PAD = 0.35
ROI_TIMEOUT_FRAMES = 20

TM_METHOD = cv.TM_CCOEFF_NORMED
TM_THRESH = 0.72
TM_SCALES = np.linspace(0.6, 1.6, 13).astype(np.float32)

ORB_NFEATURES = 800
ORB_FAST_THRESHOLD = 15
ORB_MIN_MATCHES = 12
ORB_MAX_MATCHES_DRAW = 30

FPS_AVG_N = 20

def resize_keep_ar(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img, 1.0
    s = max_w / float(w)
    nh = int(round(h * s))
    nw = int(round(w * s))
    out = cv.resize(img, (nw, nh), interpolation=cv.INTER_AREA)
    return out, s

def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def expand_box(box, pad, W, H):
    x, y, w, h = box
    cx = x + w * 0.5
    cy = y + h * 0.5
    nw = w * (1.0 + pad * 2.0)
    nh = h * (1.0 + pad * 2.0)
    nx = int(round(cx - nw * 0.5))
    ny = int(round(cy - nh * 0.5))
    return clamp_roi(nx, ny, int(round(nw)), int(round(nh)), W, H)

def draw_box(img, box, text, color):
    x, y, w, h = box
    cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
    if text:
        cv.putText(img, text, (x, max(0, y - 7)), cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv.LINE_AA)

def template_match_multiscale(gray_roi, templ_gray, scales, method):
    best_score = -1.0
    best_loc = None
    best_size = None

    H, W = gray_roi.shape[:2]

    for s in scales:
        tw = int(round(templ_gray.shape[1] * float(s)))
        th = int(round(templ_gray.shape[0] * float(s)))
        if tw < 12 or th < 12:
            continue
        if tw >= W or th >= H:
            continue

        t = cv.resize(templ_gray, (tw, th), interpolation=cv.INTER_AREA)
        res = cv.matchTemplate(gray_roi, t, method)
        _, maxv, _, maxloc = cv.minMaxLoc(res)

        if maxv > best_score:
            best_score = float(maxv)
            best_loc = maxloc
            best_size = (tw, th)

    if best_loc is None:
        return None, best_score

    x, y = best_loc
    tw, th = best_size
    return (x, y, tw, th), best_score

def orb_match_box(gray_roi, templ_gray, orb, bf):
    kp_t, des_t = orb.detectAndCompute(templ_gray, None)
    if des_t is None or kp_t is None or len(kp_t) < 4:
        return None, 0, 0, 0.0

    kp_f, des_f = orb.detectAndCompute(gray_roi, None)
    if des_f is None or kp_f is None or len(kp_f) < 4:
        return None, 0, 0, 0.0

    matches = bf.match(des_t, des_f)
    if matches is None or len(matches) < ORB_MIN_MATCHES:
        return None, len(kp_f), 0, 0.0

    matches = sorted(matches, key=lambda m: m.distance)
    good = matches[:max(ORB_MIN_MATCHES, min(len(matches), ORB_MAX_MATCHES_DRAW))]

    if len(good) < 4:
        return None, len(kp_f), len(good), 0.0

    src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    Hm, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)
    if Hm is None or mask is None:
        return None, len(kp_f), len(good), 0.0

    inliers = int(mask.ravel().sum())
    inlier_ratio = float(inliers) / float(len(good) + 1e-9)

    h, w = templ_gray.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    proj = cv.perspectiveTransform(corners, Hm).reshape(-1, 2)

    xs = proj[:, 0]
    ys = proj[:, 1]
    x0 = int(np.floor(xs.min()))
    y0 = int(np.floor(ys.min()))
    x1 = int(np.ceil(xs.max()))
    y1 = int(np.ceil(ys.max()))

    Hroi, Wroi = gray_roi.shape[:2]
    x0, y0, bw, bh = clamp_roi(x0, y0, x1 - x0, y1 - y0, Wroi, Hroi)
    if bw < 8 or bh < 8:
        return None, len(kp_f), len(good), inlier_ratio

    return (x0, y0, bw, bh), len(kp_f), len(good), inlier_ratio

def main():
    templ = cv.imread(TEMPLATE_PATH, cv.IMREAD_COLOR)
    if templ is None:
        print("Template not found:", TEMPLATE_PATH)
        return

    templ_gray = cv.cvtColor(templ, cv.COLOR_BGR2GRAY)

    cv.setUseOptimized(True)

    cap = cv.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        cap = cv.VideoCapture(CAM_INDEX, cv.CAP_V4L2)
    if not cap.isOpened():
        print("Camera not opened")
        return

    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    orb = cv.ORB_create(nfeatures=ORB_NFEATURES, fastThreshold=ORB_FAST_THRESHOLD)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    last_box = None
    last_box_age = 0

    fps_hist = deque(maxlen=FPS_AVG_N)
    t_prev = time.perf_counter()

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        frame, s = resize_keep_ar(frame, FRAME_MAX_W)
        Hf, Wf = frame.shape[:2]

        if last_box is not None and last_box_age < ROI_TIMEOUT_FRAMES:
            rx, ry, rw, rh = expand_box(last_box, ROI_PAD, Wf, Hf)
        else:
            rx, ry, rw, rh = 0, 0, Wf, Hf
            last_box = None

        roi = frame[ry:ry + rh, rx:rx + rw]
        gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

        tm_box, tm_score = template_match_multiscale(gray_roi, templ_gray, TM_SCALES, TM_METHOD)
        tm_ok = tm_box is not None and tm_score >= TM_THRESH

        orb_box, kp_count, good_count, inlier_ratio = orb_match_box(gray_roi, templ_gray, orb, bf)
        orb_ok = orb_box is not None and good_count >= ORB_MIN_MATCHES and inlier_ratio >= 0.25

        out = frame.copy()

        if rx != 0 or ry != 0 or rw != Wf or rh != Hf:
            cv.rectangle(out, (rx, ry), (rx + rw, ry + rh), (90, 90, 90), 1)

        if tm_box is not None:
            x, y, w, h = tm_box
            box_full = (x + rx, y + ry, w, h)
            txt = f"TM score={tm_score:.3f}"
            draw_box(out, box_full, txt, (0, 255, 0))

        if orb_box is not None:
            x, y, w, h = orb_box
            box_full = (x + rx, y + ry, w, h)
            txt = f"ORB inlier={inlier_ratio:.2f} good={good_count}"
            draw_box(out, box_full, txt, (255, 0, 0))

        chosen = None
        chosen_label = "NONE"
        if tm_ok and orb_ok:
            chosen = (tm_box[0] + rx, tm_box[1] + ry, tm_box[2], tm_box[3]) if tm_score >= 0.85 else (orb_box[0] + rx, orb_box[1] + ry, orb_box[2], orb_box[3])
            chosen_label = "FUSED"
        elif tm_ok:
            chosen = (tm_box[0] + rx, tm_box[1] + ry, tm_box[2], tm_box[3])
            chosen_label = "TM"
        elif orb_ok:
            chosen = (orb_box[0] + rx, orb_box[1] + ry, orb_box[2], orb_box[3])
            chosen_label = "ORB"

        if chosen is not None:
            draw_box(out, chosen, f"DETECT {chosen_label}", (0, 0, 255))
            last_box = chosen
            last_box_age = 0
        else:
            last_box_age += 1

        t1 = time.perf_counter()
        dt = (t1 - t0)
        fps = 1.0 / max(dt, 1e-9)
        fps_hist.append(fps)
        fps_avg = sum(fps_hist) / max(len(fps_hist), 1)

        latency_ms = dt * 1000.0

        cv.putText(out, f"FPS={fps_avg:.1f}  latency={latency_ms:.1f}ms  ROI={rw}x{rh}", (10, 22),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(out, f"TM_th={TM_THRESH:.2f}  kp={kp_count}  good={good_count}", (10, 46),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow("detect", out)

        k = cv.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            last_box = None
            last_box_age = ROI_TIMEOUT_FRAMES

        t_prev = t1

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
