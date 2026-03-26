import csv
import os
import time
from collections import deque
from math import hypot, sqrt

import cv2
import numpy as np

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO = None
    GPIO_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# =========================================================
# CONFIGURACION GENERAL
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720  # 720p recomendado en el PDF

# ORB + matching robusto (requisito del PDF)
RATIO_TEST = 0.75
MIN_GOOD_MATCHES_FOR_HOMOGRAPHY = 15
MIN_INLIERS_VALID = 15
MIN_INLIER_RATIO_VALID = 0.40
RANSAC_REPROJ_THRESHOLD = 3.0
MIN_HOMOGRAPHY_POINTS = 4

# Deteccion inicial por color para conservar la estructura original
MIN_AREA = 1200
MAX_AREA = 200000
EPSILON_FRAC = 0.03
ASPECT_MIN = 0.75
ASPECT_MAX = 1.30
KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Tracking
MAX_DIST = 80
MAX_MISSED = 10
TRAIL_LENGTH = 20
CENTER_HISTORY_SECONDS = 1.0

# Zona central activa 35 % x 35 % (requisito del PDF)
CENTER_ZONE_WIDTH_FRAC = 0.35
CENTER_ZONE_HEIGHT_FRAC = 0.35
CENTER_FRAMES_REQUIRED = 10

# Sesion / pruebas
TEST_DURATION_SECONDS = 90
AUTO_STOP_AFTER_TEST = True
MIN_REQUIRED_FPS = 15.0

# Archivos de salida
CAPTURE_DIR = os.path.join(BASE_DIR, "captures")
CAPTURE_COOLDOWN_SECONDS = 1.0
SESSION_CSV_PATH = os.path.join(BASE_DIR, "session.csv")
FPS_PLOT_PATH = os.path.join(BASE_DIR, "fps_vs_tiempo.png")
SESSION_SUMMARY_PATH = os.path.join(BASE_DIR, "session_summary.txt")
EVENT_LOG_PATH = os.path.join(BASE_DIR, "event_log.txt")

# Referencias por color
REF_PATHS = {
    "Rojo": os.path.join(BASE_DIR, "ref_rojo.jpg"),
    "Verde": os.path.join(BASE_DIR, "ref_verde.jpg"),
    "Azul": os.path.join(BASE_DIR, "ref_azul.jpg")
}

# Rangos HSV - conserva la logica del proyecto original
HSV_RANGES = {
    "Rojo": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255]))
    ],
    "Verde": [
        (np.array([40, 70, 60]), np.array([85, 255, 255]))
    ],
    "Azul": [
        (np.array([100, 120, 80]), np.array([125, 255, 255]))
    ]
}

# =========================================================
# GPIO + L298N
# CAMBIO PRINCIPAL: antes UART/Arduino; ahora control directo desde Raspberry Pi
# =========================================================
GPIO_USE_PWM = True
GPIO_PWM_FREQ = 1000

# Numeracion BCM
MOTOR_LEFT_IN1 = 17
MOTOR_LEFT_IN2 = 27
MOTOR_RIGHT_IN3 = 22
MOTOR_RIGHT_IN4 = 23
MOTOR_LEFT_ENA = 12
MOTOR_RIGHT_ENB = 13

# Velocidades sugeridas (0 a 100)
SPEED_FORWARD = 65
SPEED_REVERSE = 60
SPEED_TURN = 60
SPEED_SOFT_OUTER = 70
SPEED_SOFT_INNER = 35

# =========================================================
# VARIABLES GLOBALES
# =========================================================
tracked_objects = {}
next_object_id = 1

color_id_count = {
    "Rojo": 0,
    "Verde": 0,
    "Azul": 0
}

pwm_left = None
pwm_right = None
last_motion_cmd = None
simulation_mode = not GPIO_AVAILABLE

# =========================================================
# UTILIDADES DE TIEMPO / LOG
# =========================================================
def timestamp_legible():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def timestamp_archivo():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def registrar_evento(texto):
    linea = f"[{timestamp_legible()}] {texto}"
    print(linea)
    with open(EVENT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(linea + "\n")


# =========================================================
# GPIO / MOTORES
# =========================================================
def inicializar_gpio():
    """
    CAMBIO: reemplaza completamente iniciar_uart().
    Inicializa los GPIO de Raspberry Pi para controlar el puente H L298N.
    """
    global pwm_left, pwm_right, simulation_mode

    if not GPIO_AVAILABLE:
        simulation_mode = True
        print("[GPIO] RPi.GPIO no disponible. Ejecutando en modo simulacion.")
        return

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    for pin in [
        MOTOR_LEFT_IN1, MOTOR_LEFT_IN2,
        MOTOR_RIGHT_IN3, MOTOR_RIGHT_IN4,
        MOTOR_LEFT_ENA, MOTOR_RIGHT_ENB
    ]:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

    if GPIO_USE_PWM:
        pwm_left = GPIO.PWM(MOTOR_LEFT_ENA, GPIO_PWM_FREQ)
        pwm_right = GPIO.PWM(MOTOR_RIGHT_ENB, GPIO_PWM_FREQ)
        pwm_left.start(0)
        pwm_right.start(0)
    else:
        GPIO.output(MOTOR_LEFT_ENA, GPIO.HIGH)
        GPIO.output(MOTOR_RIGHT_ENB, GPIO.HIGH)

    simulation_mode = False
    print("[GPIO] Inicializacion completada. Control directo al L298N habilitado.")


def _clamp_speed(speed):
    return max(-100, min(100, int(speed)))


def _set_motor(in_a, in_b, en_pin, pwm_obj, speed):
    speed = _clamp_speed(speed)

    if simulation_mode:
        return

    if speed > 0:
        GPIO.output(in_a, GPIO.HIGH)
        GPIO.output(in_b, GPIO.LOW)
    elif speed < 0:
        GPIO.output(in_a, GPIO.LOW)
        GPIO.output(in_b, GPIO.HIGH)
    else:
        GPIO.output(in_a, GPIO.LOW)
        GPIO.output(in_b, GPIO.LOW)

    duty = abs(speed)
    if GPIO_USE_PWM and pwm_obj is not None:
        pwm_obj.ChangeDutyCycle(duty)
    else:
        GPIO.output(en_pin, GPIO.HIGH if duty > 0 else GPIO.LOW)


def aplicar_movimiento(vel_izq, vel_der):
    _set_motor(MOTOR_LEFT_IN1, MOTOR_LEFT_IN2, MOTOR_LEFT_ENA, pwm_left, vel_izq)
    _set_motor(MOTOR_RIGHT_IN3, MOTOR_RIGHT_IN4, MOTOR_RIGHT_ENB, pwm_right, vel_der)


def mover_adelante(velocidad=SPEED_FORWARD):
    aplicar_movimiento(velocidad, velocidad)


def mover_atras(velocidad=SPEED_REVERSE):
    aplicar_movimiento(-velocidad, -velocidad)


def girar_izquierda(velocidad=SPEED_TURN):
    # Giro sobre su eje: motor izquierdo atras, derecho adelante
    aplicar_movimiento(-velocidad, velocidad)


def girar_derecha(velocidad=SPEED_TURN):
    # Giro sobre su eje: motor izquierdo adelante, derecho atras
    aplicar_movimiento(velocidad, -velocidad)


def avanzar_giro_suave_izquierda(vel_ext=SPEED_SOFT_OUTER, vel_int=SPEED_SOFT_INNER):
    # Rueda izquierda mas lenta para sesgo hacia la izquierda
    aplicar_movimiento(vel_int, vel_ext)


def avanzar_giro_suave_derecha(vel_ext=SPEED_SOFT_OUTER, vel_int=SPEED_SOFT_INNER):
    # Rueda derecha mas lenta para sesgo hacia la derecha
    aplicar_movimiento(vel_ext, vel_int)


def retroceso_giro_suave_izquierda(vel_ext=SPEED_SOFT_OUTER, vel_int=SPEED_SOFT_INNER):
    aplicar_movimiento(-vel_int, -vel_ext)


def retroceso_giro_suave_derecha(vel_ext=SPEED_SOFT_OUTER, vel_int=SPEED_SOFT_INNER):
    aplicar_movimiento(-vel_ext, -vel_int)


def detener_motores():
    aplicar_movimiento(0, 0)


def ejecutar_comando_movimiento(cmd):
    """
    CAMBIO: reemplaza completamente enviar_comando_uart().
    Traduce los comandos de vision a movimientos reales sobre el L298N.
    """
    global last_motion_cmd

    if cmd == last_motion_cmd:
        return

    if cmd == "UP":
        mover_adelante()
    elif cmd == "DOWN":
        mover_atras()
    elif cmd == "LEFT":
        girar_izquierda()
    elif cmd == "RIGHT":
        girar_derecha()
    elif cmd == "LEFT_UP":
        avanzar_giro_suave_izquierda()
    elif cmd == "RIGHT_UP":
        avanzar_giro_suave_derecha()
    elif cmd == "LEFT_DOWN":
        retroceso_giro_suave_izquierda()
    elif cmd == "RIGHT_DOWN":
        retroceso_giro_suave_derecha()
    elif cmd in ("CENTER", "NO_TARGET"):
        detener_motores()
    else:
        detener_motores()

    last_motion_cmd = cmd


def limpiar_gpio():
    global pwm_left, pwm_right, last_motion_cmd

    try:
        detener_motores()
    except Exception:
        pass

    if GPIO_AVAILABLE and not simulation_mode:
        try:
            if pwm_left is not None:
                pwm_left.stop()
            if pwm_right is not None:
                pwm_right.stop()
        finally:
            GPIO.cleanup()

    pwm_left = None
    pwm_right = None
    last_motion_cmd = None


# =========================================================
# FEATURES / MATCHING
# =========================================================
def crear_detector_y_matcher():
    detector = cv2.ORB_create(
        nfeatures=1500,
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=15
    )
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return detector, matcher


def extraer_features(detector, imagen_gray):
    kp, des = detector.detectAndCompute(imagen_gray, None)
    return kp, des


def clasificar_estado_metricas(inliers, inlier_ratio, good_matches):
    if good_matches == 0:
        return "SIN_MATCH"
    if inliers >= MIN_INLIERS_VALID and inlier_ratio >= MIN_INLIER_RATIO_VALID:
        return "ROBUSTO"
    if inliers >= 8 and inlier_ratio >= 0.30:
        return "ACEPTABLE"
    if inliers >= 4:
        return "DEBIL"
    return "SIN_HOMOGRAFIA"


def _validar_poligono(poligono):
    if poligono is None or len(poligono) != 4:
        return False

    area = cv2.contourArea(poligono.astype(np.float32))
    if area < 100.0:
        return False

    return cv2.isContourConvex(poligono.reshape(-1, 1, 2))


def obtener_metricas_match(detector, matcher, referencia, roi_gray, roi_offset_xy):
    metrics = {
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "homography_ok": False,
        "status": "SIN_HOMOGRAFIA",
        "center": None,
        "polygon": None,
        "kp_roi": [],
        "des_roi": None,
        "good_matches_list": [],
        "inlier_mask": None,
        "roi_gray": roi_gray,
        "homography": None
    }

    kp_ref = referencia["kp"]
    des_ref = referencia["des"]
    gray_ref = referencia["gray"]

    kp_roi, des_roi = extraer_features(detector, roi_gray)
    metrics["kp_roi"] = kp_roi if kp_roi is not None else []
    metrics["des_roi"] = des_roi

    if gray_ref is None or des_ref is None or des_roi is None:
        return metrics

    if kp_ref is None or kp_roi is None:
        return metrics

    if len(des_ref) < 2 or len(des_roi) < 2:
        return metrics

    matches_knn = matcher.knnMatch(des_ref, des_roi, k=2)

    good = []
    for pair in matches_knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < RATIO_TEST * n.distance:
                good.append(m)

    metrics["good_matches"] = len(good)
    metrics["good_matches_list"] = good

    if len(good) < max(MIN_HOMOGRAPHY_POINTS, MIN_GOOD_MATCHES_FOR_HOMOGRAPHY):
        metrics["status"] = clasificar_estado_metricas(0, 0.0, len(good))
        return metrics

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_roi[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    try:
        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            RANSAC_REPROJ_THRESHOLD
        )
    except cv2.error:
        H, mask = None, None

    if H is None or mask is None:
        metrics["status"] = clasificar_estado_metricas(0, 0.0, len(good))
        return metrics

    inliers = int(mask.ravel().sum())
    inlier_ratio = inliers / float(len(good)) if good else 0.0

    metrics["inliers"] = inliers
    metrics["inlier_ratio"] = inlier_ratio
    metrics["inlier_mask"] = mask.ravel().astype(bool)
    metrics["homography"] = H
    metrics["status"] = clasificar_estado_metricas(inliers, inlier_ratio, len(good))

    h_ref, w_ref = gray_ref.shape[:2]
    ref_corners = np.float32([
        [0, 0],
        [w_ref - 1, 0],
        [w_ref - 1, h_ref - 1],
        [0, h_ref - 1]
    ]).reshape(-1, 1, 2)

    try:
        projected_roi = cv2.perspectiveTransform(ref_corners, H)
        projected_full = projected_roi + np.float32([[[roi_offset_xy[0], roi_offset_xy[1]]]])
        polygon = np.int32(np.round(projected_full)).reshape(-1, 2)
    except cv2.error:
        polygon = None

    if polygon is not None and _validar_poligono(polygon):
        center = tuple(np.mean(polygon, axis=0).astype(int))
        metrics["center"] = center
        metrics["polygon"] = polygon

    if (
        metrics["center"] is not None and
        metrics["polygon"] is not None and
        inliers >= MIN_INLIERS_VALID and
        inlier_ratio >= MIN_INLIER_RATIO_VALID
    ):
        metrics["homography_ok"] = True

    return metrics


# =========================================================
# REFERENCIAS
# =========================================================
def cargar_referencias(detector):
    referencias = {}

    for color, path in REF_PATHS.items():
        img = cv2.imread(path)

        if img is None:
            print(f"[ADVERTENCIA] No se pudo cargar la referencia para {color}: {path}")
            referencias[color] = {
                "img": None,
                "gray": None,
                "kp": None,
                "des": None,
                "path": path
            }
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = extraer_features(detector, gray)

        referencias[color] = {
            "img": img,
            "gray": gray,
            "kp": kp,
            "des": des,
            "path": path
        }

        print(f"[REF] {color}: {len(kp) if kp is not None else 0} keypoints cargados")

    return referencias


# =========================================================
# DETECCION POR HSV + VALIDACION CON ORB + HOMOGRAFIA
# =========================================================
def detectar_objetos(frame_bgr, detector, matcher, referencias):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    detections = []

    for color, ranges in HSV_RANGES.items():
        referencia = referencias.get(color)
        if referencia is None or referencia["gray"] is None:
            continue

        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in ranges:
            mask = cv2.inRange(hsv, low, high)
            mask_total = cv2.bitwise_or(mask_total, mask)

        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, KERNEL)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, KERNEL)

        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA or area > MAX_AREA:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, EPSILON_FRAC * peri, True)
            x, y, w, h = cv2.boundingRect(approx)

            if h == 0:
                continue

            aspect = w / float(h)
            if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
                continue

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame_bgr.shape[1], x + w)
            y2 = min(frame_bgr.shape[0], y + h)

            roi_gray = gray[y1:y2, x1:x2]
            if roi_gray.size == 0:
                continue

            metricas = obtener_metricas_match(
                detector=detector,
                matcher=matcher,
                referencia=referencia,
                roi_gray=roi_gray,
                roi_offset_xy=(x1, y1)
            )

            cx = x + w // 2
            cy = y + h // 2
            center = metricas["center"] if metricas["center"] is not None else (cx, cy)

            detections.append({
                "color": color,
                "bbox": (x, y, w, h),
                "center": center,
                "raw_center": (cx, cy),
                "matches": metricas["good_matches"],
                "inliers": metricas["inliers"],
                "inlier_ratio": metricas["inlier_ratio"],
                "status": metricas["status"],
                "contour": approx,
                "homography_ok": metricas["homography_ok"],
                "polygon": metricas["polygon"],
                "kp_roi": metricas["kp_roi"],
                "des_roi": metricas["des_roi"],
                "good_matches_list": metricas["good_matches_list"],
                "inlier_mask": metricas["inlier_mask"],
                "roi_gray": metricas["roi_gray"],
                "roi_offset": (x1, y1),
                "reference_img": referencia["img"],
                "reference_kp": referencia["kp"]
            })

    return detections


# =========================================================
# TRACKING
# =========================================================
def generar_label(color):
    global color_id_count

    color_id_count[color] += 1
    if color == "Rojo":
        return f"R{color_id_count[color]}"
    if color == "Verde":
        return f"G{color_id_count[color]}"
    if color == "Azul":
        return f"B{color_id_count[color]}"
    return f"OBJ{color_id_count[color]}"


def _actualizar_historial_centro(obj, center, timestamp_s):
    history = obj["center_history"]
    history.append((timestamp_s, center))

    while history and (timestamp_s - history[0][0]) > CENTER_HISTORY_SECONDS:
        history.popleft()

    if len(history) < 2:
        obj["rms_center_1s"] = 0.0
        return

    xs = np.array([p[1][0] for p in history], dtype=np.float32)
    ys = np.array([p[1][1] for p in history], dtype=np.float32)
    mx = xs.mean()
    my = ys.mean()
    rms = sqrt(np.mean((xs - mx) ** 2 + (ys - my) ** 2))
    obj["rms_center_1s"] = float(rms)


def registrar_objeto(detection, timestamp_s):
    global tracked_objects, next_object_id

    obj_id = next_object_id
    next_object_id += 1

    tracked_objects[obj_id] = {
        "label": generar_label(detection["color"]),
        "color": detection["color"],
        "bbox": detection["bbox"],
        "center": detection["center"],
        "raw_center": detection["raw_center"],
        "missed": 0,
        "matches": detection["matches"],
        "inliers": detection["inliers"],
        "inlier_ratio": detection["inlier_ratio"],
        "status": detection["status"],
        "trail": deque([detection["center"]], maxlen=TRAIL_LENGTH),
        "center_history": deque([(timestamp_s, detection["center"])]),
        "rms_center_1s": 0.0,
        "homography_ok": detection["homography_ok"],
        "polygon": detection["polygon"],
        "kp_roi": detection["kp_roi"],
        "des_roi": detection["des_roi"],
        "good_matches_list": detection["good_matches_list"],
        "inlier_mask": detection["inlier_mask"],
        "roi_gray": detection["roi_gray"],
        "roi_offset": detection["roi_offset"],
        "reference_img": detection["reference_img"],
        "reference_kp": detection["reference_kp"]
    }


def eliminar_objeto(obj_id):
    global tracked_objects
    if obj_id in tracked_objects:
        del tracked_objects[obj_id]


def actualizar_tracking(detections, timestamp_s):
    global tracked_objects

    if len(detections) == 0:
        ids_borrar = []
        for obj_id in list(tracked_objects.keys()):
            tracked_objects[obj_id]["missed"] += 1
            if tracked_objects[obj_id]["missed"] > MAX_MISSED:
                ids_borrar.append(obj_id)

        for obj_id in ids_borrar:
            eliminar_objeto(obj_id)

        return tracked_objects

    if len(tracked_objects) == 0:
        for det in detections:
            registrar_objeto(det, timestamp_s)
        return tracked_objects

    used_detections = set()
    object_ids = list(tracked_objects.keys())

    for obj_id in object_ids:
        obj = tracked_objects[obj_id]

        best_det_idx = -1
        best_dist = float("inf")

        for i, det in enumerate(detections):
            if i in used_detections:
                continue
            if det["color"] != obj["color"]:
                continue

            d = hypot(
                det["center"][0] - obj["center"][0],
                det["center"][1] - obj["center"][1]
            )

            if d < best_dist:
                best_dist = d
                best_det_idx = i

        if best_det_idx != -1 and best_dist < MAX_DIST:
            det = detections[best_det_idx]
            obj["bbox"] = det["bbox"]
            obj["center"] = det["center"]
            obj["raw_center"] = det["raw_center"]
            obj["missed"] = 0
            obj["matches"] = det["matches"]
            obj["inliers"] = det["inliers"]
            obj["inlier_ratio"] = det["inlier_ratio"]
            obj["status"] = det["status"]
            obj["trail"].append(det["center"])
            obj["homography_ok"] = det["homography_ok"]
            obj["polygon"] = det["polygon"]
            obj["kp_roi"] = det["kp_roi"]
            obj["des_roi"] = det["des_roi"]
            obj["good_matches_list"] = det["good_matches_list"]
            obj["inlier_mask"] = det["inlier_mask"]
            obj["roi_gray"] = det["roi_gray"]
            obj["roi_offset"] = det["roi_offset"]
            obj["reference_img"] = det["reference_img"]
            obj["reference_kp"] = det["reference_kp"]
            _actualizar_historial_centro(obj, det["center"], timestamp_s)
            used_detections.add(best_det_idx)
        else:
            obj["missed"] += 1

    ids_borrar = []
    for obj_id in list(tracked_objects.keys()):
        if tracked_objects[obj_id]["missed"] > MAX_MISSED:
            ids_borrar.append(obj_id)

    for obj_id in ids_borrar:
        eliminar_objeto(obj_id)

    for i, det in enumerate(detections):
        if i not in used_detections:
            registrar_objeto(det, timestamp_s)

    return tracked_objects


# =========================================================
# OBJETIVO, ESTADOS Y COMANDO DE CENTRADO
# =========================================================
def obtener_limites_zona_central(frame_shape):
    h, w = frame_shape[:2]
    zone_w = int(w * CENTER_ZONE_WIDTH_FRAC)
    zone_h = int(h * CENTER_ZONE_HEIGHT_FRAC)

    x1 = (w - zone_w) // 2
    y1 = (h - zone_h) // 2
    x2 = x1 + zone_w
    y2 = y1 + zone_h

    return x1, y1, x2, y2


def seleccionar_objetivo(objetos, frame_shape):
    if len(objetos) == 0:
        return None

    h, w = frame_shape[:2]
    cx_frame = w // 2
    cy_frame = h // 2

    candidatos = []
    for obj_id, obj in objetos.items():
        if obj["missed"] > 0:
            continue
        if obj["inliers"] < MIN_INLIERS_VALID or obj["inlier_ratio"] < MIN_INLIER_RATIO_VALID:
            continue
        if not obj["homography_ok"]:
            continue

        dx = obj["center"][0] - cx_frame
        dy = obj["center"][1] - cy_frame
        dist = hypot(dx, dy)

        score = (
            obj["inliers"],
            obj["inlier_ratio"],
            obj["matches"],
            -obj["rms_center_1s"],
            -dist
        )
        candidatos.append((score, obj_id))

    if not candidatos:
        return None

    candidatos.sort(key=lambda item: item[0], reverse=True)
    return candidatos[0][1]


def calcular_comando_centrado(center_obj, frame_shape):
    """
    Regresa:
    - texto para pantalla
    - comando de movimiento
    - dx, dy
    - centrado (True/False)
    """
    h, w = frame_shape[:2]
    cx_frame = w // 2
    cy_frame = h // 2

    x1, y1, x2, y2 = obtener_limites_zona_central(frame_shape)
    tol_x = (x2 - x1) // 2
    tol_y = (y2 - y1) // 2

    dx = center_obj[0] - cx_frame
    dy = center_obj[1] - cy_frame

    if dx < -tol_x:
        horiz = "LEFT"
        txt_h = "IZQUIERDA"
    elif dx > tol_x:
        horiz = "RIGHT"
        txt_h = "DERECHA"
    else:
        horiz = ""
        txt_h = ""

    if dy < -tol_y:
        vert = "UP"
        txt_v = "ARRIBA"
    elif dy > tol_y:
        vert = "DOWN"
        txt_v = "ABAJO"
    else:
        vert = ""
        txt_v = ""

    if horiz == "" and vert == "":
        return "CENTRADO", "CENTER", dx, dy, True

    if horiz != "" and vert != "":
        texto = f"MOVER {txt_h} | {txt_v}"
        cmd = f"{horiz}_{vert}"
    elif horiz != "":
        texto = f"MOVER {txt_h}"
        cmd = horiz
    else:
        texto = f"MOVER {txt_v}"
        cmd = vert

    return texto, cmd, dx, dy, False


# =========================================================
# DIBUJO / HUD
# =========================================================
def color_bgr(nombre):
    if nombre == "Rojo":
        return (0, 0, 255)
    if nombre == "Verde":
        return (0, 255, 0)
    if nombre == "Azul":
        return (255, 0, 0)
    return (255, 255, 255)


def contar_por_color(objetos):
    conteo = {"Rojo": 0, "Verde": 0, "Azul": 0}
    for _, obj in objetos.items():
        conteo[obj["color"]] += 1
    return conteo


def dibujar_zona_central(frame, objetivo_en_zona=False):
    h, w = frame.shape[:2]
    cx = w // 2
    cy = h // 2
    x1, y1, x2, y2 = obtener_limites_zona_central(frame.shape)

    color = (0, 255, 0) if objetivo_en_zona else (0, 255, 255)

    cv2.drawMarker(
        frame,
        (cx, cy),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=30,
        thickness=2
    )

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        "ZONA CENTRAL 35% x 35%",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )


def construir_overlay_matches(objetivo, max_width=360, max_height=220):
    ref_img = objetivo.get("reference_img")
    kp_ref = objetivo.get("reference_kp")
    roi_gray = objetivo.get("roi_gray")
    kp_roi = objetivo.get("kp_roi")
    good_matches = objetivo.get("good_matches_list") or []
    inlier_mask = objetivo.get("inlier_mask")

    if ref_img is None or roi_gray is None or kp_ref is None or kp_roi is None:
        return None

    if len(good_matches) == 0:
        return None

    roi_bgr = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

    try:
        preview = cv2.drawMatches(
            ref_img,
            kp_ref,
            roi_bgr,
            kp_roi,
            good_matches,
            None,
            matchesMask=inlier_mask.tolist() if inlier_mask is not None else None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
    except cv2.error:
        return None

    h, w = preview.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    preview = cv2.resize(preview, new_size, interpolation=cv2.INTER_AREA)
    return preview


def incrustar_preview(frame, preview, x=20, y=500):
    if preview is None:
        return

    ph, pw = preview.shape[:2]
    fh, fw = frame.shape[:2]

    if y + ph > fh:
        y = max(0, fh - ph - 10)
    if x + pw > fw:
        x = max(0, fw - pw - 10)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 4, y - 24), (x + pw + 4, y + ph + 4), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
    frame[y:y + ph, x:x + pw] = preview
    cv2.rectangle(frame, (x - 4, y - 24), (x + pw + 4, y + ph + 4), (255, 255, 255), 1)
    cv2.putText(frame, "Matches / Inliers", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def dibujar_hud(frame, objetivo, fps, avg_fps, elapsed_s, estado_hud, centered_frames):
    x0, y0 = 20, 140
    x1, y1 = 440, 370

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 1)

    if objetivo is None:
        etiqueta = "-"
        inliers = 0
        inlier_ratio = 0.0
        matches = 0
        rms_center = 0.0
    else:
        etiqueta = objetivo["label"]
        inliers = objetivo["inliers"]
        inlier_ratio = objetivo["inlier_ratio"]
        matches = objetivo["matches"]
        rms_center = objetivo.get("rms_center_1s", 0.0)

    lineas = [
        f"Estado: {estado_hud}",
        f"Objetivo: {etiqueta}",
        f"Tiempo: {elapsed_s:6.1f}/{TEST_DURATION_SECONDS} s",
        f"FPS: {fps:.2f}",
        f"FPS promedio: {avg_fps:.2f}",
        f"Matches: {matches}",
        f"Inliers: {inliers}",
        f"Inlier-ratio: {inlier_ratio:.2f}",
        f"RMS centro (1 s): {rms_center:.2f} px",
        f"Frames centrado: {centered_frames}/{CENTER_FRAMES_REQUIRED}"
    ]

    y = y0 + 26
    for linea in lineas:
        cv2.putText(
            frame,
            linea,
            (x0 + 12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2
        )
        y += 22


def dibujar_resultados(frame, objetos, objetivo_id, instruccion, mov_cmd, dx, dy,
                       fps, avg_fps, elapsed_s, estado_hud, centered_frames):
    h, w = frame.shape[:2]
    cx_frame = w // 2
    cy_frame = h // 2

    objetivo = objetos[objetivo_id] if objetivo_id is not None else None
    objetivo_en_zona = objetivo is not None and mov_cmd == "CENTER"

    dibujar_zona_central(frame, objetivo_en_zona)

    for obj_id, obj in objetos.items():
        x, y, w_box, h_box = obj["bbox"]
        c = color_bgr(obj["color"])
        grosor = 4 if obj_id == objetivo_id else 2

        if obj.get("polygon") is not None:
            cv2.polylines(frame, [obj["polygon"]], True, c, grosor)
        else:
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), c, grosor)

        texto = f'{obj["label"]} {obj["color"]} M:{obj["matches"]} I:{obj["inliers"]} R:{obj["inlier_ratio"]:.2f}'
        if obj_id == objetivo_id:
            texto += " [OBJETIVO]"

        cv2.putText(
            frame,
            texto,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            c,
            2
        )

        radio = 6 if obj_id == objetivo_id else 4
        cv2.circle(frame, obj["center"], radio, c, -1)

        pts = list(obj["trail"])
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], c, 2)

    if objetivo_id is not None:
        cv2.line(frame, (cx_frame, cy_frame), objetivo["center"], (0, 255, 255), 2)
        color_texto = (0, 255, 0) if mov_cmd == "CENTER" else (0, 0, 255)

        cv2.putText(frame, f"Instruccion: {instruccion}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.90, color_texto, 2)
        cv2.putText(frame, f"Movimiento: {mov_cmd}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 0), 2)
        cv2.putText(frame, f"Error X: {dx:+d} px   Error Y: {dy:+d} px", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 0), 2)

        preview = construir_overlay_matches(objetivo)
        incrustar_preview(frame, preview)
    else:
        cv2.putText(frame, "Instruccion: SIN OBJETIVO", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2)
        cv2.putText(frame, "Movimiento: NO_TARGET", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 0), 2)

    dibujar_hud(frame, objetivo, fps, avg_fps, elapsed_s, estado_hud, centered_frames)

    conteo = contar_por_color(objetos)
    total = conteo["Rojo"] + conteo["Verde"] + conteo["Azul"]
    cv2.putText(frame,
                f"#R: {conteo['Rojo']}  #G: {conteo['Verde']}  #B: {conteo['Azul']}  #Total: {total}",
                (20, 395),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 2)
    cv2.putText(frame, "Tecla: q=Salir", (20, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (150, 150, 150), 2)


# =========================================================
# REGISTRO, CAPTURAS Y REPORTE
# =========================================================
def preparar_directorios():
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    with open(EVENT_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("=== EVENT LOG ===\n")


def crear_csv_session(path_csv):
    csv_file = open(path_csv, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    # Formato obligatorio del PDF
    writer.writerow([
        "timestamp",
        "fps",
        "inliers",
        "inlier_ratio",
        "cx",
        "cy",
        "estado",
        "ruta_captura"
    ])
    return csv_file, writer


def generar_nombre_captura(objetivo):
    ts = timestamp_archivo()
    color = objetivo.get("color", "NA") if objetivo is not None else "NA"
    inliers = objetivo.get("inliers", 0) if objetivo is not None else 0
    ratio = objetivo.get("inlier_ratio", 0.0) if objetivo is not None else 0.0
    nombre = f"{ts}_{color}_{inliers:03d}_{ratio:.2f}.jpg".replace(" ", "_")
    return os.path.join(CAPTURE_DIR, nombre)


def guardar_captura_automatica(frame, objetivo):
    path = generar_nombre_captura(objetivo)
    ok = cv2.imwrite(path, frame)
    if ok:
        registrar_evento(f"CAPTURA guardada: {path}")
        return path
    registrar_evento(f"ERROR al guardar captura: {path}")
    return ""


def registrar_frame_csv(writer, fps, objetivo, estado, capture_path):
    if objetivo is None:
        inliers = 0
        inlier_ratio = 0.0
        cx = ""
        cy = ""
    else:
        inliers = objetivo["inliers"]
        inlier_ratio = objetivo["inlier_ratio"]
        cx = objetivo["center"][0]
        cy = objetivo["center"][1]

    writer.writerow([
        timestamp_legible(),
        f"{fps:.4f}",
        int(inliers),
        f"{inlier_ratio:.4f}",
        cx,
        cy,
        estado,
        capture_path
    ])


def generar_grafica_fps(times_s, fps_values, output_path):
    if plt is None:
        print("[REPORTE] matplotlib no esta instalado. No se genero la grafica de FPS.")
        return False

    if len(times_s) == 0 or len(fps_values) == 0:
        print("[REPORTE] No hay datos para generar la grafica de FPS.")
        return False

    plt.figure(figsize=(10, 4.5))
    plt.plot(times_s, fps_values)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("FPS")
    plt.title("FPS vs tiempo")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[REPORTE] Grafica generada: {output_path}")
    return True


def guardar_resumen_sesion(path_summary, elapsed_s, total_frames, avg_fps,
                           captures_count, avg_recovery_time, recovery_events):
    cumple_fps = avg_fps >= MIN_REQUIRED_FPS
    cumple_recovery = (avg_recovery_time is not None and avg_recovery_time < 2.0)

    with open(path_summary, "w", encoding="utf-8") as f:
        f.write("=== RESUMEN DE SESION ===\n")
        f.write(f"Duracion registrada: {elapsed_s:.2f} s\n")
        f.write(f"Frames procesados: {total_frames}\n")
        f.write(f"FPS promedio: {avg_fps:.2f}\n")
        f.write(f"Capturas automaticas: {captures_count}\n")
        f.write(f"Resolucion configurada: {FRAME_WIDTH}x{FRAME_HEIGHT}\n")
        f.write(f"Meta FPS >= {MIN_REQUIRED_FPS:.0f}: {'CUMPLE' if cumple_fps else 'NO CUMPLE'}\n")
        if recovery_events > 0:
            f.write(f"Eventos de recuperacion: {recovery_events}\n")
            f.write(f"Tiempo promedio de recuperacion: {avg_recovery_time:.3f} s\n")
            f.write(f"Meta recuperacion < 2 s: {'CUMPLE' if cumple_recovery else 'NO CUMPLE'}\n")
        else:
            f.write("Eventos de recuperacion: 0\n")
            f.write("Tiempo promedio de recuperacion: N/A\n")

        f.write("\nGPIO usados para L298N:\n")
        f.write(f"IN1={MOTOR_LEFT_IN1}, IN2={MOTOR_LEFT_IN2}, IN3={MOTOR_RIGHT_IN3}, IN4={MOTOR_RIGHT_IN4}\n")
        f.write(f"ENA={MOTOR_LEFT_ENA}, ENB={MOTOR_RIGHT_ENB}, PWM={'SI' if GPIO_USE_PWM else 'NO'}\n")

        if not cumple_fps:
            f.write("Nota: si no se alcanza 15 FPS, documenta en el reporte la optimizacion aplicada.\n")

    print(f"[REPORTE] Resumen guardado: {path_summary}")


# =========================================================
# MAIN
# =========================================================
def main():
    global last_motion_cmd

    detector, matcher = crear_detector_y_matcher()
    referencias = cargar_referencias(detector)

    preparar_directorios()
    registrar_evento("Inicio de sesion")

    inicializar_gpio()  # CAMBIO: antes iniciar_uart()
    csv_file, csv_writer = crear_csv_session(SESSION_CSV_PATH)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        csv_file.close()
        limpiar_gpio()
        raise RuntimeError("No se pudo abrir la camara")

    prev_time = time.time()
    session_start = prev_time
    fps_hist = []
    time_hist = []
    frame_idx = 0
    centered_consecutive_frames = 0
    last_capture_time = 0.0
    captures_count = 0
    prev_state = "BUSCANDO"
    prev_target_id = None
    loss_started_at = None
    recovery_times = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                registrar_evento("Lectura de camara fallida. Fin de sesion.")
                break

            frame_idx += 1
            loop_time = time.time()
            elapsed_s = loop_time - session_start

            detections = detectar_objetos(frame, detector, matcher, referencias)
            objetos = actualizar_tracking(detections, elapsed_s)
            objetivo_id = seleccionar_objetivo(objetos, frame.shape)

            if objetivo_id is not None:
                objetivo = objetos[objetivo_id]
                instruccion, mov_cmd, dx, dy, centrado = calcular_comando_centrado(
                    objetivo["center"], frame.shape
                )
            else:
                objetivo = None
                instruccion = "SIN OBJETIVO"
                mov_cmd = "NO_TARGET"
                dx, dy = 0, 0
                centrado = False

            if objetivo is None:
                estado_hud = "BUSCANDO"
                centered_consecutive_frames = 0
            else:
                if centrado:
                    centered_consecutive_frames += 1
                else:
                    centered_consecutive_frames = 0

                if centered_consecutive_frames >= CENTER_FRAMES_REQUIRED:
                    estado_hud = "CENTRADO"
                    mov_cmd = "CENTER"
                    instruccion = f"CENTRADO ({centered_consecutive_frames}/{CENTER_FRAMES_REQUIRED})"
                else:
                    estado_hud = "SIGUIENDO"

            # Requisito de perdida de objetivo: inliers < 15 o ratio < 0.40 => BUSCANDO
            if objetivo is None or objetivo["inliers"] < MIN_INLIERS_VALID or objetivo["inlier_ratio"] < MIN_INLIER_RATIO_VALID:
                estado_hud = "BUSCANDO"
                mov_cmd = "NO_TARGET"
                instruccion = "BUSCANDO OBJETIVO"
                centered_consecutive_frames = 0

            ejecutar_comando_movimiento(mov_cmd)  # CAMBIO: antes enviar_comando_uart(...)

            curr_time = time.time()
            delta = curr_time - prev_time
            fps = 1.0 / delta if delta > 0 else 0.0
            prev_time = curr_time

            elapsed_s = curr_time - session_start
            fps_hist.append(fps)
            time_hist.append(elapsed_s)
            avg_fps = sum(fps_hist) / len(fps_hist)

            # Logica de perdida / recuperacion
            if prev_state != "BUSCANDO" and estado_hud == "BUSCANDO":
                registrar_evento(
                    f"PERDIDA objetivo={prev_target_id} inliers={0 if objetivo is None else objetivo['inliers']} "
                    f"ratio={0.0 if objetivo is None else objetivo['inlier_ratio']:.2f}"
                )
                loss_started_at = curr_time

            if prev_state == "BUSCANDO" and estado_hud in ("SIGUIENDO", "CENTRADO"):
                if loss_started_at is not None:
                    recovery_time = curr_time - loss_started_at
                    recovery_times.append(recovery_time)
                    registrar_evento(f"RECUPERACION en {recovery_time:.3f} s")
                    loss_started_at = None
                else:
                    registrar_evento("OBJETIVO ADQUIRIDO")

            dibujar_resultados(
                frame, objetos, objetivo_id, instruccion, mov_cmd, dx, dy,
                fps, avg_fps, elapsed_s, estado_hud, centered_consecutive_frames
            )

            capture_path = ""
            if (
                estado_hud == "CENTRADO" and
                centered_consecutive_frames == CENTER_FRAMES_REQUIRED and
                (curr_time - last_capture_time) >= CAPTURE_COOLDOWN_SECONDS
            ):
                capture_path = guardar_captura_automatica(frame, objetivo)
                if capture_path:
                    captures_count += 1
                    last_capture_time = curr_time

            registrar_frame_csv(csv_writer, fps, objetivo, estado_hud, capture_path)

            cv2.imshow("Tracking ORB + GPIO Raspberry Pi", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                registrar_evento("Salida solicitada por teclado")
                break

            if AUTO_STOP_AFTER_TEST and elapsed_s >= TEST_DURATION_SECONDS:
                registrar_evento(f"Se alcanzaron {TEST_DURATION_SECONDS} segundos de prueba")
                break

            prev_state = estado_hud
            prev_target_id = objetivo_id

    finally:
        final_elapsed = time.time() - session_start
        avg_fps = (sum(fps_hist) / len(fps_hist)) if fps_hist else 0.0
        avg_recovery_time = (sum(recovery_times) / len(recovery_times)) if recovery_times else None

        csv_file.flush()
        csv_file.close()
        generar_grafica_fps(time_hist, fps_hist, FPS_PLOT_PATH)
        guardar_resumen_sesion(
            SESSION_SUMMARY_PATH,
            final_elapsed,
            frame_idx,
            avg_fps,
            captures_count,
            avg_recovery_time,
            len(recovery_times)
        )

        print(f"[CSV] Registro guardado en: {SESSION_CSV_PATH}")
        print(f"[CSV] Frames registrados: {frame_idx}")
        print(f"[CSV] FPS promedio: {avg_fps:.2f}")
        if avg_fps < MIN_REQUIRED_FPS:
            print("[ADVERTENCIA] El promedio quedo por debajo de 15 FPS. Debes justificar tecnicamente la optimizacion en el reporte.")

        cap.release()
        limpiar_gpio()
        cv2.destroyAllWindows()
        registrar_evento("Fin de sesion")


if __name__ == "__main__":
    main()
