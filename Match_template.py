import os
import csv
import cv2
import time
import math
import numpy as np
from datetime import datetime
from collections import deque

try:
    import serial
except Exception:
    serial = None

# =========================
# CONFIGURACION GENERAL
# =========================
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720   # 720p recomendado
SERIAL_PORT = None             # ejemplo: 'COM5' en Windows, '/dev/ttyUSB0' en Linux
BAUDRATE = 115200

TEMPLATE_DIR = "templates"
CAPTURE_DIR = "captures"
LOG_CSV = "session.csv"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# ORB / geometria
NFEATURES = 1200
RATIO_TEST = 0.75
RANSAC_REPROJ_THR = 3.0
MIN_GOOD_MATCHES = 15
MIN_INLIERS = 15
MIN_INLIER_RATIO = 0.40

# color + estabilidad
MIN_COLOR_AREA = 2500
VALID_FRAMES_TO_LOCK = 3
CENTER_FRAMES_TO_CAPTURE = 10
CENTER_ZONE_SCALE = 0.35
CAPTURE_COOLDOWN_S = 1.5
EMA_ALPHA = 0.35
CENTER_HISTORY_LEN = 15

# HSV base (misma idea que tu entrega 1, pero ahora se usa como ROI para ORB)
COLOR_RANGES = {
    "red": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([179, 255, 255]))
    ],
    "green": [
        (np.array([35, 80, 60]), np.array([85, 255, 255]))
    ],
    "blue": [
        (np.array([95, 80, 60]), np.array([130, 255, 255]))
    ],
    "white": [
        (np.array([0, 0, 200]), np.array([179, 60, 255]))
    ]
}

MOVE_BY_COLOR = {
    "white": "WHITE",
    "red": "RED",
    "green": "GREEN",
    "blue": "BLUE",
    "none": "STOP"
}

TEMPLATE_PATHS = {
    "white": os.path.join(TEMPLATE_DIR, "white.png"),
    "red": os.path.join(TEMPLATE_DIR, "red.png"),
    "green": os.path.join(TEMPLATE_DIR, "green.png"),
    "blue": os.path.join(TEMPLATE_DIR, "blue.png"),
}


# =========================
# UTILIDADES
# =========================
def now_stamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def build_masks(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    masks = {}
    for color, ranges in COLOR_RANGES.items():
        mask = None
        for lo, hi in ranges:
            part = cv2.inRange(hsv, lo, hi)
            mask = part if mask is None else cv2.bitwise_or(mask, part)
        masks[color] = clean_mask(mask)
    return masks


def largest_contour_bbox(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area <= 0:
        return None, 0.0
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, w, h), area


def polygon_is_valid(poly, frame_shape):
    h, w = frame_shape[:2]
    if poly is None or poly.shape != (4, 1, 2):
        return False

    pts = poly.reshape(4, 2)
    if not np.isfinite(pts).all():
        return False

    if np.any(pts[:, 0] < -50) or np.any(pts[:, 0] > w + 50):
        return False
    if np.any(pts[:, 1] < -50) or np.any(pts[:, 1] > h + 50):
        return False

    area = abs(cv2.contourArea(pts.astype(np.float32)))
    if area < 1200:
        return False

    hull = cv2.convexHull(pts.astype(np.float32))
    if len(hull) != 4:
        return False

    sides = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        sides.append(np.linalg.norm(p2 - p1))
    sides = np.array(sides, dtype=np.float32)
    if np.min(sides) < 12:
        return False

    ratio = float(np.max(sides) / max(np.min(sides), 1e-6))
    if ratio > 6.0:  # evita deformaciones absurdas
        return False

    return True


def make_center_zone(frame_w, frame_h, scale=CENTER_ZONE_SCALE):
    zw = int(frame_w * scale)
    zh = int(frame_h * scale)
    x1 = (frame_w - zw) // 2
    y1 = (frame_h - zh) // 2
    x2 = x1 + zw
    y2 = y1 + zh
    return (x1, y1, x2, y2)


def point_in_rect(pt, rect):
    if pt is None:
        return False
    x, y = pt
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def rms_center_step(center_history):
    if len(center_history) < 2:
        return 0.0
    diffs = []
    pts = list(center_history)
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        diffs.append(dx * dx + dy * dy)
    return float(math.sqrt(np.mean(diffs))) if diffs else 0.0


def draw_hud(frame, state, fps, best, center_zone, centered_frames, rms_px, sent_cmd):
    out = frame.copy()
    x1, y1, x2, y2 = center_zone
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(out, "ZONA CENTRAL 35% x 35%", (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)

    panel = [
        f"Estado: {state}",
        f"FPS: {fps:5.2f}",
        f"Cmd serial: {sent_cmd}",
    ]

    if best is None:
        panel += [
            "Color: none",
            "Good matches: 0",
            "Inliers: 0",
            "Inlier-ratio: 0.00",
            f"Center RMS: {rms_px:5.2f}px",
            f"Centered frames: {centered_frames}",
        ]
    else:
        panel += [
            f"Color: {best['color']}",
            f"Area color ROI: {int(best['area'])}",
            f"Good matches: {best['good_matches']}",
            f"Inliers: {best['inliers']}",
            f"Inlier-ratio: {best['inlier_ratio']:.2f}",
            f"Center RMS: {rms_px:5.2f}px",
            f"Centered frames: {centered_frames}",
        ]

        poly = best.get("projected_bbox")
        if poly is not None:
            cv2.polylines(out, [np.int32(poly)], True, (0, 255, 255), 3)

        c = best.get("center_smooth")
        if c is not None:
            cx, cy = int(c[0]), int(c[1])
            cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(out, f"C=({cx},{cy})", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    y = 28
    for text in panel:
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28

    return out


def build_matches_view(template_img, kp_t, frame_bgr, kp_f, good_matches, inlier_mask):
    if template_img is None or kp_t is None or frame_bgr is None or kp_f is None or not good_matches:
        return np.zeros((240, 640, 3), dtype=np.uint8)

    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    matches_mask = inlier_mask.tolist() if inlier_mask is not None else None
    vis = cv2.drawMatches(
        template_img, kp_t,
        frame_bgr, kp_f,
        good_matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        matchesMask=matches_mask,
        flags=flags
    )
    vis = cv2.resize(vis, (min(1280, vis.shape[1]), 360))
    return vis


class SerialCommander:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.last_sent = "STOP"
        self.last_send_t = 0.0
        if serial is not None and port:
            try:
                self.ser = serial.Serial(port, baudrate, timeout=0)
                time.sleep(2.0)
                print(f"[Serial] conectado en {port} @ {baudrate}")
            except Exception as e:
                print(f"[Serial] no se pudo abrir {port}: {e}")
                self.ser = None
        else:
            print("[Serial] desactivado (SERIAL_PORT=None o pyserial no instalado)")

    def send(self, cmd):
        cmd = (cmd or "STOP").strip().upper()
        now = time.time()
        if cmd == self.last_sent and (now - self.last_send_t) < 0.20:
            return self.last_sent
        if self.ser is not None:
            try:
                self.ser.write((cmd + "\n").encode("utf-8"))
            except Exception as e:
                print(f"[Serial] error al enviar {cmd}: {e}")
        self.last_sent = cmd
        self.last_send_t = now
        return self.last_sent

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass


# =========================
# CARGA DE PLANTILLAS
# =========================
def load_templates(orb):
    templates = {}
    for color, path in TEMPLATE_PATHS.items():
        if not os.path.exists(path):
            print(f"[WARN] falta plantilla: {path}")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] no pude leer plantilla: {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if des is None or len(kp) < MIN_GOOD_MATCHES:
            print(f"[WARN] plantilla '{color}' con muy pocas features ORB ({0 if kp is None else len(kp)}).")
            continue
        h, w = gray.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        templates[color] = {
            "img": img,
            "gray": gray,
            "kp": kp,
            "des": des,
            "corners": corners,
            "size": (w, h)
        }
        print(f"[OK] plantilla cargada: {color} -> {path} ({len(kp)} kp)")
    return templates


# =========================
# DETECCION POR COLOR + ORB
# =========================
def detect_best_target(frame_bgr, gray, masks, templates, orb, bf):
    best = None
    best_debug = None

    for color, tpl in templates.items():
        mask = masks.get(color)
        if mask is None:
            continue

        bbox, area = largest_contour_bbox(mask)
        if bbox is None or area < MIN_COLOR_AREA:
            continue

        kp_f, des_f = orb.detectAndCompute(gray, mask)
        if des_f is None or kp_f is None or len(kp_f) < MIN_GOOD_MATCHES:
            continue

        try:
            knn = bf.knnMatch(tpl["des"], des_f, k=2)
        except Exception:
            continue

        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < RATIO_TEST * n.distance:
                good.append(m)

        if len(good) < MIN_GOOD_MATCHES:
            continue

        src_pts = np.float32([tpl["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THR)
        if H is None or inlier_mask is None:
            continue

        inlier_mask = inlier_mask.ravel().astype(bool)
        inliers = int(np.sum(inlier_mask))
        inlier_ratio = float(inliers / max(len(good), 1))

        projected = cv2.perspectiveTransform(tpl["corners"], H)
        if not polygon_is_valid(projected, frame_bgr.shape):
            continue

        center = projected.reshape(4, 2).mean(axis=0)

        candidate = {
            "color": color,
            "bbox_color": bbox,
            "area": area,
            "good_matches": len(good),
            "inliers": inliers,
            "inlier_ratio": inlier_ratio,
            "projected_bbox": projected,
            "center_raw": center,
            "H": H,
            "kp_frame": kp_f,
            "good_matches_list": good,
            "inlier_mask": inlier_mask,
        }

        valid = (inliers >= MIN_INLIERS) and (inlier_ratio >= MIN_INLIER_RATIO)
        candidate["valid"] = valid
        if not valid:
            continue

        if best is None:
            best = candidate
            best_debug = (tpl, kp_f, good, inlier_mask)
        else:
            if (candidate["inliers"] > best["inliers"]) or \
               (candidate["inliers"] == best["inliers"] and candidate["inlier_ratio"] > best["inlier_ratio"]):
                best = candidate
                best_debug = (tpl, kp_f, good, inlier_mask)

    return best, best_debug


# =========================
# CSV
# =========================
def init_csv(path):
    file_exists = os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "datetime_iso", "elapsed_s", "fps", "state", "color",
            "area_color_roi", "good_matches", "inliers", "inlier_ratio",
            "cx", "cy", "centered", "centered_frames", "rms_center_px",
            "recovery_s", "sent_command", "capture_file"
        ])
        f.flush()
    return f, writer


# =========================
# MAIN
# =========================
def main():
    orb = cv2.ORB_create(nfeatures=NFEATURES, fastThreshold=12)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    templates = load_templates(orb)
    if not templates:
        print("No hay plantillas válidas. Crea /templates/white.png, red.png, green.png y blue.png")
        return

    ser_cmd = SerialCommander(SERIAL_PORT, BAUDRATE)
    csv_file, csv_writer = init_csv(LOG_CSV)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("No pude abrir la cámara.")
        csv_file.close()
        ser_cmd.close()
        return

    center_zone = make_center_zone(FRAME_W, FRAME_H)

    state = "BUSCANDO"
    valid_frames = 0
    centered_frames = 0
    center_history = deque(maxlen=CENTER_HISTORY_LEN)
    smooth_center = None
    last_capture_t = 0.0
    loss_start_t = None
    last_time = time.time()
    t0 = last_time

    print("Sistema listo. Teclas: q=salir | s=forzar STOP")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        now = time.time()
        dt = max(now - last_time, 1e-6)
        fps = 1.0 / dt
        last_time = now
        elapsed = now - t0
        capture_file = ""
        recovery_s = ""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masks = build_masks(frame)
        best, best_debug = detect_best_target(frame, gray, masks, templates, orb, bf)

        prev_state = state

        if best is None:
            state = "BUSCANDO"
            valid_frames = 0
            centered_frames = 0
            center_history.clear()
            smooth_center = None
            if prev_state != "BUSCANDO":
                loss_start_t = now
            sent_cmd = ser_cmd.send("STOP")
            rms_px = 0.0
            matches_view = np.zeros((360, 960, 3), dtype=np.uint8)
            cv2.putText(matches_view, "Sin deteccion valida (ORB / RANSAC)", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            if smooth_center is None:
                smooth_center = best["center_raw"].astype(np.float32)
            else:
                smooth_center = (EMA_ALPHA * best["center_raw"] + (1.0 - EMA_ALPHA) * smooth_center).astype(np.float32)
            best["center_smooth"] = smooth_center.copy()
            center_history.append(smooth_center.copy())
            rms_px = rms_center_step(center_history)

            if prev_state == "BUSCANDO" and loss_start_t is not None:
                recovery_s = round(now - loss_start_t, 3)
                loss_start_t = None

            valid_frames += 1
            is_centered = point_in_rect(best["center_smooth"], center_zone)
            if is_centered:
                centered_frames += 1
                state = "CENTRADO"
            else:
                centered_frames = 0
                state = "SIGUIENDO"

            # solo manda comando cuando ya se estabilizó la detección
            if valid_frames >= VALID_FRAMES_TO_LOCK:
                sent_cmd = ser_cmd.send(MOVE_BY_COLOR.get(best["color"], "STOP"))
            else:
                sent_cmd = ser_cmd.send("STOP")

            if centered_frames >= CENTER_FRAMES_TO_CAPTURE and (now - last_capture_t) > CAPTURE_COOLDOWN_S:
                filename = (
                    f"{best['color']}_{now_stamp()}"
                    f"_inl{best['inliers']}_ir{best['inlier_ratio']:.2f}.jpg"
                )
                capture_file = os.path.join(CAPTURE_DIR, filename)
                preview_tmp = draw_hud(frame, state, fps, best, center_zone, centered_frames, rms_px, sent_cmd)
                cv2.imwrite(capture_file, preview_tmp)
                print(f"[CAPTURA] {capture_file}")
                last_capture_t = now
                centered_frames = 0

            tpl, kp_f, good, inlier_mask = best_debug
            matches_view = build_matches_view(tpl["img"], tpl["kp"], frame, kp_f, good, inlier_mask)

        preview = draw_hud(frame, state, fps, best, center_zone, centered_frames, rms_px, sent_cmd)
        cv2.imshow("Vision robusta ORB + color + HUD", preview)
        cv2.imshow("Matches / Inliers", matches_view)

        # CSV por frame
        if best is None:
            csv_writer.writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                round(elapsed, 3), round(fps, 3), state, "none",
                0, 0, 0, 0.0,
                "", "", 0, centered_frames, round(rms_px, 3),
                recovery_s, sent_cmd, capture_file
            ])
        else:
            cx, cy = best["center_smooth"]
            csv_writer.writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                round(elapsed, 3), round(fps, 3), state, best["color"],
                int(best["area"]), best["good_matches"], best["inliers"], round(best["inlier_ratio"], 4),
                round(float(cx), 2), round(float(cy), 2),
                int(point_in_rect(best["center_smooth"], center_zone)), centered_frames,
                round(rms_px, 3), recovery_s, sent_cmd, capture_file
            ])
        csv_file.flush()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ser_cmd.send("STOP")

    ser_cmd.send("STOP")
    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    ser_cmd.close()


if __name__ == "__main__":
    main()
