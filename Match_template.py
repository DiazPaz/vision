#seguimiento

import os
import csv
import cv2
import time
import numpy as np
from datetime import datetime
from collections import deque

try:
    import serial
except Exception:
    serial = None

# ==================== CONFIG ====================
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720

# Imagen objetivo para ORB (la que SIEMPRE debe seguir)
TEMPLATE_PATH = "target.png"

# Serial hacia Arduino (tu Arduino original se conserva)
USE_SERIAL = True
SERIAL_PORT = "COM5"      # CAMBIA esto por tu puerto real
SERIAL_BAUD = 115200

# ORB / matching
ORB_FEATURES = 1500
RATIO_TEST = 0.75
MIN_GOOD_MATCHES = 15
MIN_INLIERS = 15
MIN_INLIER_RATIO = 0.40
RANSAC_REPROJ_THRESH = 3.0
MIN_PROJECTED_AREA = 1800

# Estabilidad
STABLE_FRAMES = 4
LOST_FRAMES_TO_SEARCH = 3
CAPTURE_FRAMES = 10
CAPTURE_COOLDOWN_S = 1.5
COMMAND_COOLDOWN_S = 0.10

# Control de seguimiento
CENTER_ZONE_RATIO = 0.20     # 20% del ancho para considerar centrado en X
TARGET_AREA = 30000          # ajustar con pruebas reales
AREA_TOL = 0.22              # tolerancia +-22%
EMA_ALPHA = 0.35
CENTER_RMS_N = 10

SAVE_DIR = "captures_target"
CSV_PATH = "session.csv"
os.makedirs(SAVE_DIR, exist_ok=True)

# Se mantiene compatibilidad con tu Arduino original
SERIAL_MAP = {
    "FORWARD": "WHITE",
    "BACKWARD": "RED",
    "RIGHT": "GREEN",
    "LEFT": "BLUE",
    "STOP": "STOP",
}

# ==================== UTILS ====================
def now_stamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def rms_deviation(points):
    if len(points) < 2:
        return 0.0
    pts = np.array(points, dtype=np.float32)
    mean = np.mean(pts, axis=0)
    diff = pts - mean
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def open_serial():
    if not USE_SERIAL:
        print("[SERIAL] Desactivado")
        return None
    if serial is None:
        print("[SERIAL] pyserial no instalado, seguire sin Arduino")
        return None
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.01)
        time.sleep(2.0)
        print(f"[SERIAL] Conectado en {SERIAL_PORT} @ {SERIAL_BAUD}")
        return ser
    except Exception as e:
        print(f"[SERIAL] No pude abrir {SERIAL_PORT}: {e}")
        return None


def send_command(ser, command, cache):
    serial_cmd = SERIAL_MAP[command]
    t = time.time()
    if serial_cmd != cache["last_cmd"] or (t - cache["last_send_t"] >= COMMAND_COOLDOWN_S):
        if ser is not None:
            try:
                ser.write((serial_cmd + "\n").encode("utf-8"))
            except Exception as e:
                print(f"[SERIAL] Error enviando comando: {e}")
        cache["last_cmd"] = serial_cmd
        cache["last_send_t"] = t
    return serial_cmd


def init_csv(path):
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if new_file:
        writer.writerow([
            "timestamp", "elapsed_s", "fps", "state", "good_matches", "inliers",
            "inlier_ratio", "center_x", "center_y", "smooth_x", "smooth_y",
            "projected_area", "smooth_area", "center_rms_px", "valid_streak",
            "centered_streak", "command", "serial_cmd"
        ])
    return f, writer


# ==================== TEMPLATE ====================
def load_template(path, orb):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(
            f"No encontre '{path}'. Pon la imagen objetivo en la misma carpeta del script."
        )

    kp, des = orb.detectAndCompute(img, None)
    if des is None or len(kp) < 8:
        raise RuntimeError(
            "La plantilla tiene muy poca textura para ORB. Usa una imagen con esquinas y detalles."
        )

    h, w = img.shape[:2]
    return img, kp, des, (w, h)


# ==================== ORB DETECTION ====================
def detect_target(frame_bgr, template_data, orb, matcher):
    tpl_img, tpl_kp, tpl_des, (tpl_w, tpl_h) = template_data
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(gray, None)

    fallback_vis = np.hstack([
        cv2.cvtColor(tpl_img, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ])

    if des2 is None or len(kp2) < 8:
        return {
            "valid": False,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "quad": None,
            "bbox": None,
            "center": None,
            "projected_area": 0.0,
            "matches_vis": fallback_vis,
        }

    knn = matcher.knnMatch(tpl_des, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO_TEST * n.distance:
            good.append(m)

    if len(good) < MIN_GOOD_MATCHES:
        vis = cv2.drawMatches(
            tpl_img, tpl_kp, frame_bgr, kp2, good[:60], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return {
            "valid": False,
            "good_matches": len(good),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "quad": None,
            "bbox": None,
            "center": None,
            "projected_area": 0.0,
            "matches_vis": vis,
        }

    src_pts = np.float32([tpl_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH)
    if H is None or mask is None:
        vis = cv2.drawMatches(
            tpl_img, tpl_kp, frame_bgr, kp2, good[:60], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return {
            "valid": False,
            "good_matches": len(good),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "quad": None,
            "bbox": None,
            "center": None,
            "projected_area": 0.0,
            "matches_vis": vis,
        }

    matches_mask = mask.ravel().tolist()
    inliers = int(np.sum(mask))
    inlier_ratio = float(inliers / max(len(good), 1))

    corners = np.float32([[0, 0], [tpl_w - 1, 0], [tpl_w - 1, tpl_h - 1], [0, tpl_h - 1]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    x_min = int(np.min(quad[:, 0]))
    y_min = int(np.min(quad[:, 1]))
    x_max = int(np.max(quad[:, 0]))
    y_max = int(np.max(quad[:, 1]))
    bbox = (x_min, y_min, x_max, y_max)

    projected_area = float(abs(cv2.contourArea(quad.astype(np.float32))))
    cx = int(np.mean(quad[:, 0]))
    cy = int(np.mean(quad[:, 1]))

    h, w = frame_bgr.shape[:2]
    inside_margin = 60
    valid = (
        inliers >= MIN_INLIERS and
        inlier_ratio >= MIN_INLIER_RATIO and
        projected_area >= MIN_PROJECTED_AREA and
        np.all(quad[:, 0] >= -inside_margin) and np.all(quad[:, 0] <= w + inside_margin) and
        np.all(quad[:, 1] >= -inside_margin) and np.all(quad[:, 1] <= h + inside_margin)
    )

    vis = cv2.drawMatches(
        tpl_img, tpl_kp, frame_bgr, kp2, good[:60], None,
        matchesMask=matches_mask[:60] if len(matches_mask) >= len(good[:60]) else None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return {
        "valid": bool(valid),
        "good_matches": len(good),
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "quad": quad.astype(np.int32),
        "bbox": bbox,
        "center": (cx, cy),
        "projected_area": projected_area,
        "matches_vis": vis,
    }


# ==================== FOLLOW LOGIC ====================
def decide_follow_command(sx, smooth_area, frame_w):
    center_x = frame_w // 2
    dead_zone = int(frame_w * CENTER_ZONE_RATIO / 2.0)
    error_x = sx - center_x

    # 1) primero centra el objeto
    if error_x < -dead_zone:
        return "LEFT"
    if error_x > dead_zone:
        return "RIGHT"

    # 2) ya centrado, regula distancia para seguirlo
    min_area = TARGET_AREA * (1.0 - AREA_TOL)
    max_area = TARGET_AREA * (1.0 + AREA_TOL)

    if smooth_area < min_area:
        return "FORWARD"
    if smooth_area > max_area:
        return "BACKWARD"

    # 3) si ya esta centrado y a buena distancia, se queda quieto
    return "STOP"


# ==================== DRAW ====================
def annotate(frame, det, state, fps, valid_streak, centered_streak,
             smooth_center, smooth_area, center_rms, command):
    out = frame.copy()
    h, w = out.shape[:2]

    # zona central visual
    dead_zone = int(w * CENTER_ZONE_RATIO / 2.0)
    cx0 = w // 2
    cv2.line(out, (cx0, 0), (cx0, h), (255, 255, 0), 1)
    cv2.rectangle(out, (cx0 - dead_zone, 0), (cx0 + dead_zone, h), (0, 255, 255), 2)

    if det["valid"] and det["quad"] is not None:
        cv2.polylines(out, [det["quad"]], True, (0, 255, 0), 3)
        cx, cy = det["center"]
        cv2.circle(out, (cx, cy), 6, (255, 0, 255), -1)
        cv2.putText(out, "OBJETIVO", (max(0, cx - 45), max(0, cy - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    if smooth_center is not None:
        sx, sy = int(smooth_center[0]), int(smooth_center[1])
        cv2.circle(out, (sx, sy), 8, (0, 255, 255), 2)
        cv2.line(out, (w // 2, h // 2), (sx, sy), (0, 255, 255), 2)

    info = [
        f"Estado: {state}",
        f"FPS: {fps:.1f}",
        f"Good matches: {det['good_matches']}",
        f"Inliers: {det['inliers']}",
        f"Inlier-ratio: {det['inlier_ratio']:.2f}",
        f"Area proj: {det['projected_area']:.0f}",
        f"Area smooth: {smooth_area:.0f}",
        f"Stable: {valid_streak}",
        f"Centered frames: {centered_streak}",
        f"Center RMS: {center_rms:.1f}px",
        f"Comando: {command}",
    ]

    y = 28
    for line in info:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28

    return out


# ==================== MAIN ====================
def main():
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    try:
        template_data = load_template(TEMPLATE_PATH, orb)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    ser = open_serial()
    csv_file, csv_writer = init_csv(CSV_PATH)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("No pude abrir la camara.")
        csv_file.close()
        if ser is not None:
            ser.close()
        return

    serial_cache = {"last_cmd": None, "last_send_t": 0.0}
    prev_t = time.time()
    t0 = prev_t
    last_save_t = 0.0

    valid_streak = 0
    lost_frames = 0
    centered_streak = 0
    state = "BUSCANDO"
    command = "STOP"

    smooth_center = None
    smooth_area = 0.0
    center_history = deque(maxlen=CENTER_RMS_N)

    print("Listo. Presiona 'q' para salir.")
    print("Si la camara ve el objeto de target.png, el carro lo seguira continuamente.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        now_t = time.time()
        dt = max(now_t - prev_t, 1e-6)
        fps = 1.0 / dt
        prev_t = now_t
        elapsed_s = now_t - t0

        det = detect_target(frame, template_data, orb, matcher)

        if det["valid"]:
            lost_frames = 0
            valid_streak += 1

            cx, cy = det["center"]
            area = det["projected_area"]

            if smooth_center is None:
                smooth_center = (float(cx), float(cy))
                smooth_area = float(area)
            else:
                smooth_center = (
                    EMA_ALPHA * cx + (1.0 - EMA_ALPHA) * smooth_center[0],
                    EMA_ALPHA * cy + (1.0 - EMA_ALPHA) * smooth_center[1],
                )
                smooth_area = EMA_ALPHA * area + (1.0 - EMA_ALPHA) * smooth_area

            center_history.append(smooth_center)
            center_rms = rms_deviation(center_history)
            sx, sy = int(smooth_center[0]), int(smooth_center[1])

            desired_cmd = "STOP"
            if valid_streak >= STABLE_FRAMES:
                desired_cmd = decide_follow_command(sx, smooth_area, frame.shape[1])

            command = desired_cmd
            serial_cmd = send_command(ser, command, serial_cache)

            center_x = frame.shape[1] // 2
            dead_zone = int(frame.shape[1] * CENTER_ZONE_RATIO / 2.0)
            centered_now = abs(sx - center_x) <= dead_zone
            if centered_now:
                centered_streak += 1
            else:
                centered_streak = 0

            if command in ("LEFT", "RIGHT", "FORWARD", "BACKWARD"):
                state = "SIGUIENDO"
            else:
                state = "ALINEADO"

            if centered_streak >= CAPTURE_FRAMES and (now_t - last_save_t > CAPTURE_COOLDOWN_S):
                filename = (
                    f"target_{state.lower()}_{now_stamp()}_"
                    f"in{det['inliers']}_r{det['inlier_ratio']:.2f}.jpg"
                )
                path = os.path.join(SAVE_DIR, filename)
                preview = annotate(frame, det, state, fps, valid_streak, centered_streak,
                                   smooth_center, smooth_area, center_rms, command)
                cv2.imwrite(path, preview)
                print(f"[CAPTURA] Guardada: {path}")
                last_save_t = now_t

        else:
            lost_frames += 1
            valid_streak = 0
            centered_streak = 0
            smooth_center = None
            smooth_area = 0.0
            center_history.clear()
            center_rms = 0.0

            if lost_frames >= LOST_FRAMES_TO_SEARCH:
                state = "BUSCANDO"
                command = "STOP"
            serial_cmd = send_command(ser, command, serial_cache)

        preview = annotate(frame, det, state, fps, valid_streak, centered_streak,
                           smooth_center, smooth_area, center_rms, command)

        csv_writer.writerow([
            datetime.now().isoformat(timespec="milliseconds"),
            f"{elapsed_s:.3f}",
            f"{fps:.3f}",
            state,
            det["good_matches"],
            det["inliers"],
            f"{det['inlier_ratio']:.3f}",
            "" if det["center"] is None else det["center"][0],
            "" if det["center"] is None else det["center"][1],
            "" if smooth_center is None else f"{smooth_center[0]:.1f}",
            "" if smooth_center is None else f"{smooth_center[1]:.1f}",
            f"{det['projected_area']:.1f}",
            f"{smooth_area:.1f}",
            f"{center_rms:.3f}",
            valid_streak,
            centered_streak,
            command,
            serial_cmd,
        ])
        csv_file.flush()

        cv2.imshow("Seguimiento del objeto (ORB)", preview)
        cv2.imshow("Matches e Inliers", det["matches_vis"])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    send_command(ser, "STOP", serial_cache)
    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    if ser is not None:
        ser.close()


if __name__ == "__main__":
    main()
