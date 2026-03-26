"""
=============================================================================
  VISIÓN COMPUTACIONAL + CONTROL GPIO  –  Raspberry Pi  (sin Arduino/UART)
  Proyecto Final  |  Segundo Entregable  |  Departamento de Ingeniería UDEM
=============================================================================

CAMBIOS RESPECTO A LA VERSIÓN UART
───────────────────────────────────
  ELIMINADO:
    • import serial / pyserial
    • Constantes SERIAL_PORT, SERIAL_BAUD, SERIAL_TIMEOUT, SERIAL_RESEND_INTERVAL
    • Variables globales last_uart_cmd / last_uart_time
    • Funciones iniciar_uart() y enviar_comando_uart()
    • Objeto 'ser' y su cierre en el bloque finally

  AÑADIDO (sección marcada con [GPIO]):
    • import RPi.GPIO
    • Constantes de pines GPIO_IN1 … GPIO_ENB
    • Funciones: inicializar_gpio(), mover_adelante(), mover_atras(),
      girar_izquierda(), girar_derecha(), detener_motores(),
      ejecutar_comando_movimiento(), limpiar_gpio()
    • Lógica de velocidad diferencial para comandos diagonales

  VISIÓN: sin cambios funcionales; solo se renombró "uart_cmd" → "cmd_gpio"
          en el HUD y en el CSV para reflejar que ya no hay UART.
=============================================================================
"""

# ── Importaciones estándar ──────────────────────────────────────────────────
import csv
import os
import time
from collections import deque
from math import hypot

import cv2
import numpy as np

# ── [GPIO] Importar RPi.GPIO ────────────────────────────────────────────────
# Se importa con try/except para que el código no falle en una PC de desarrollo
try:
    import RPi.GPIO as GPIO
    GPIO_DISPONIBLE = True
except ImportError:
    GPIO = None
    GPIO_DISPONIBLE = False
    print("[GPIO] RPi.GPIO no encontrado. Modo simulación (sin movimiento real).")

# Matplotlib es opcional (solo para la gráfica de FPS al final)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================
CAM_INDEX      = 0
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 360        # 720p recomendado por el entregable

RATIO_TEST     = 0.75       # Ratio test de Lowe

MIN_AREA       = 1200
MAX_AREA       = 200000
EPSILON_FRAC   = 0.03
ASPECT_MIN     = 0.75
ASPECT_MAX     = 1.30

MAX_DIST       = 80
MAX_MISSED     = 10
TRAIL_LENGTH   = 20

# Tolerancia para considerar el objeto centrado (píxeles)
CENTER_TOL_X   = 50
CENTER_TOL_Y   = 50

MIN_HOMOGRAPHY_POINTS    = 4
RANSAC_REPROJ_THRESHOLD  = 5.0

# Sesión / prueba
TEST_DURATION_SECONDS = 90
AUTO_STOP_AFTER_TEST  = True
MIN_REQUIRED_FPS      = 15.0

# Capturas y archivos de sesión
CAPTURE_DIR          = "captures"
CAPTURE_COOLDOWN_SECONDS = 1.0
SESSION_CSV_PATH     = "session.csv"
FPS_PLOT_PATH        = "fps_vs_tiempo.png"
SESSION_SUMMARY_PATH = "session_summary.txt"


# =============================================================================
# [GPIO] PINES DEL L298N
# =============================================================================
#
#  Raspberry Pi  ──────────────────────────────────►  L298N
#  ─────────────────────────────────────────────────────────
#  GPIO 17  (BCM)  →  IN1   (dirección motor izquierdo A)
#  GPIO 27  (BCM)  →  IN2   (dirección motor izquierdo B)
#  GPIO 12  (BCM)  →  ENA   (enable motor izquierdo – PWM hardware)
#
#  GPIO 23  (BCM)  →  IN3   (dirección motor derecho A)
#  GPIO 24  (BCM)  →  IN4   (dirección motor derecho B)
#  GPIO 13  (BCM)  →  ENB   (enable motor derecho – PWM hardware)
#
#  NOTA: GPIO 12 y GPIO 13 son los dos pines de PWM hardware del RPi.
#        Si prefieres software PWM, cualquier pin GPIO sirve, pero
#        GPIO 12/13 dan señal más limpia y sin carga de CPU extra.
#
#  PRECAUCIONES ELÉCTRICAS (ver sección al final del archivo):
#    - Alimenta el L298N con 6-12 V DC externos; NUNCA desde el RPi.
#    - Conecta las tierras (GND) del RPi y del L298N.
#    - No conectes motores de >600 mA directamente sin disipador en el L298N.
#    - Nivel lógico del RPi es 3.3 V; el L298N acepta desde 2.3 V → compatible.
#
GPIO_IN1 = 17   # Motor izquierdo – dirección A
GPIO_IN2 = 27   # Motor izquierdo – dirección B
GPIO_ENA = 12   # Motor izquierdo – enable (PWM)

GPIO_IN3 = 23   # Motor derecho – dirección A
GPIO_IN4 = 24   # Motor derecho – dirección B
GPIO_ENB = 13   # Motor derecho – enable (PWM)

# Frecuencia del PWM (Hz) y velocidades
PWM_FREQ          = 1000    # 1 kHz es suficiente para motores DC genéricos
VELOCIDAD_NORMAL  = 40      # % duty cycle para movimientos puros  (0-100)
VELOCIDAD_DIAGONAL_RAPIDO = 40   # motor "dominante" en diagonal
VELOCIDAD_DIAGONAL_LENTO  = 25   # motor "secundario" en diagonal (giro suave)

# Objetos PWM globales (se inicializan en inicializar_gpio)
_pwm_ena = None
_pwm_enb = None


# =============================================================================
# [GPIO] FUNCIONES DE CONTROL DE MOTORES
# =============================================================================

def inicializar_gpio():
    """
    [GPIO – NUEVO]
    Configura todos los pines del L298N como salidas y arranca el PWM.
    Llama a esta función UNA sola vez al inicio del programa.
    """
    global _pwm_ena, _pwm_enb

    if not GPIO_DISPONIBLE:
        print("[GPIO] Simulación: inicializar_gpio() no hace nada.")
        return

    GPIO.setmode(GPIO.BCM)          # usamos numeración BCM
    GPIO.setwarnings(False)

    # Configurar pines de dirección como salida
    for pin in (GPIO_IN1, GPIO_IN2, GPIO_IN3, GPIO_IN4):
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)  # arrancar en LOW = motores detenidos

    # Configurar pines de enable como salida y crear PWM
    GPIO.setup(GPIO_ENA, GPIO.OUT)
    GPIO.setup(GPIO_ENB, GPIO.OUT)

    _pwm_ena = GPIO.PWM(GPIO_ENA, PWM_FREQ)
    _pwm_enb = GPIO.PWM(GPIO_ENB, PWM_FREQ)

    _pwm_ena.start(0)   # arrancar con duty 0 (motor parado)
    _pwm_enb.start(0)

    print(f"[GPIO] Pines configurados. ENA={GPIO_ENA}, ENB={GPIO_ENB}, "
          f"IN1={GPIO_IN1}, IN2={GPIO_IN2}, IN3={GPIO_IN3}, IN4={GPIO_IN4}")


def _set_motor_izquierdo(adelante: bool, velocidad: float):
    """Pone el motor izquierdo en dirección y velocidad dadas."""
    if not GPIO_DISPONIBLE:
        return
    if adelante:
        GPIO.output(GPIO_IN1, GPIO.HIGH)
        GPIO.output(GPIO_IN2, GPIO.LOW)
    else:
        GPIO.output(GPIO_IN1, GPIO.LOW)
        GPIO.output(GPIO_IN2, GPIO.HIGH)
    _pwm_ena.ChangeDutyCycle(velocidad)


def _set_motor_derecho(adelante: bool, velocidad: float):
    """Pone el motor derecho en dirección y velocidad dadas."""
    if not GPIO_DISPONIBLE:
        return
    if adelante:
        GPIO.output(GPIO_IN3, GPIO.HIGH)
        GPIO.output(GPIO_IN4, GPIO.LOW)
    else:
        GPIO.output(GPIO_IN3, GPIO.LOW)
        GPIO.output(GPIO_IN4, GPIO.HIGH)
    _pwm_enb.ChangeDutyCycle(velocidad)


def mover_adelante(velocidad=VELOCIDAD_NORMAL):
    """[GPIO] Ambos motores hacia adelante (comando UP)."""
    if not GPIO_DISPONIBLE:
        print("[GPIO-SIM] mover_adelante()")
        return
    _set_motor_izquierdo(False,  velocidad)
    _set_motor_derecho(False,    velocidad)


def mover_atras(velocidad=VELOCIDAD_NORMAL):
    """[GPIO] Ambos motores hacia atrás (comando DOWN)."""
    if not GPIO_DISPONIBLE:
        print("[GPIO-SIM] mover_atras()")
        return
    _set_motor_izquierdo(True, velocidad)
    _set_motor_derecho(True,   velocidad)


def girar_izquierda(velocidad=VELOCIDAD_NORMAL):
    """
    [GPIO] Giro en su sitio a la izquierda (comando LEFT).
    Motor derecho adelante, motor izquierdo atrás.
    """
    if not GPIO_DISPONIBLE:
        print("[GPIO-SIM] girar_izquierda()")
        return
    _set_motor_izquierdo(False, velocidad)
    _set_motor_derecho(True,    velocidad)


def girar_derecha(velocidad=VELOCIDAD_NORMAL):
    """
    [GPIO] Giro en su sitio a la derecha (comando RIGHT).
    Motor izquierdo adelante, motor derecho atrás.
    """
    if not GPIO_DISPONIBLE:
        print("[GPIO-SIM] girar_derecha()")
        return
    _set_motor_izquierdo(True,  velocidad)
    _set_motor_derecho(False,   velocidad)


def arco_izquierda_adelante():
    """
    [GPIO] Arco curvo hacia adelante-izquierda (comando LEFT_UP).
    Motor derecho rápido, motor izquierdo lento → el carro curva a la izquierda
    mientras avanza.
    """
    if not GPIO_DISPONIBLE:
        print("[GPIO-SIM] arco_izquierda_adelante()")
        return
    _set_motor_izquierdo(True, VELOCIDAD_DIAGONAL_LENTO)
    _set_motor_derecho(True,   VELOCIDAD_DIAGONAL_RAPIDO)


def arco_derecha_adelante():
    """
    [GPIO] Arco curvo hacia adelante-derecha (comando RIGHT_UP).
    Motor izquierdo rápido, motor derecho lento → el carro curva a la derecha
    mientras avanza.
    """
    if not GPIO_DISPONIBLE:
        print("[GPIO-SIM] arco_derecha_adelante()")
        return
    _set_motor_izquierdo(True, VELOCIDAD_DIAGONAL_RAPIDO)
    _set_motor_derecho(True,   VELOCIDAD_DIAGONAL_LENTO)


def arco_izquierda_atras():
    """
    [GPIO] Arco curvo hacia atrás-izquierda (comando LEFT_DOWN).
    Ambos motores en reversa; el derecho más rápido.
    """
    if not GPIO_DISPONIBLE:
        print("[GPIO-SIM] arco_izquierda_atras()")
        return
    _set_motor_izquierdo(False, VELOCIDAD_DIAGONAL_LENTO)
    _set_motor_derecho(False,   VELOCIDAD_DIAGONAL_RAPIDO)


def arco_derecha_atras():
    """
    [GPIO] Arco curvo hacia atrás-derecha (comando RIGHT_DOWN).
    Ambos motores en reversa; el izquierdo más rápido.
    """
    if not GPIO_DISPONIBLE:
        print("[GPIO-SIM] arco_derecha_atras()")
        return
    _set_motor_izquierdo(False, VELOCIDAD_DIAGONAL_RAPIDO)
    _set_motor_derecho(False,   VELOCIDAD_DIAGONAL_LENTO)


def detener_motores():
    """[GPIO] Detiene ambos motores (comando CENTER o NO_TARGET)."""
    if not GPIO_DISPONIBLE:
        print("[GPIO-SIM] detener_motores()")
        return
    for pin in (GPIO_IN1, GPIO_IN2, GPIO_IN3, GPIO_IN4):
        GPIO.output(pin, GPIO.LOW)
    _pwm_ena.ChangeDutyCycle(0)
    _pwm_enb.ChangeDutyCycle(0)


def ejecutar_comando_movimiento(cmd: str):
    """
    [GPIO – NUEVO, reemplaza enviar_comando_uart()]
    Traduce el comando de visión a acción real sobre los motores.

    Tabla de comandos:
    ┌──────────────┬────────────────────────────────────────────┐
    │  Comando     │  Movimiento del carro                      │
    ├──────────────┼────────────────────────────────────────────┤
    │  UP          │  Avanzar recto                             │
    │  DOWN        │  Retroceder recto                          │
    │  LEFT        │  Girar en sitio a la izquierda             │
    │  RIGHT       │  Girar en sitio a la derecha               │
    │  LEFT_UP     │  Arco adelante-izquierda (vel. diferencial)│
    │  RIGHT_UP    │  Arco adelante-derecha   (vel. diferencial)│
    │  LEFT_DOWN   │  Arco atrás-izquierda    (vel. diferencial)│
    │  RIGHT_DOWN  │  Arco atrás-derecha      (vel. diferencial)│
    │  CENTER      │  Detener (objeto centrado)                 │
    │  NO_TARGET   │  Detener (sin objetivo detectado)          │
    └──────────────┴────────────────────────────────────────────┘

    Decisión sobre diagonales: se usa velocidad diferencial (un motor
    más lento en lugar de parar) para hacer un arco suave en lugar de
    un giro brusco en su sitio. Esto es más robusto para sistemas de
    seguimiento en tiempo real donde el objeto puede estar levemente
    desplazado en dos ejes simultáneamente.
    """
    if cmd == "UP":
        mover_adelante()
    elif cmd == "DOWN":
        mover_atras()
    elif cmd == "LEFT":
        girar_izquierda()
    elif cmd == "RIGHT":
        girar_derecha()
    elif cmd == "LEFT_UP":
        arco_izquierda_adelante()
    elif cmd == "RIGHT_UP":
        arco_derecha_adelante()
    elif cmd == "LEFT_DOWN":
        arco_izquierda_atras()
    elif cmd == "RIGHT_DOWN":
        arco_derecha_atras()
    elif cmd in ("CENTER", "NO_TARGET"):
        detener_motores()
    else:
        # Comando desconocido: parar por seguridad
        detener_motores()


def limpiar_gpio():
    """
    [GPIO] Detiene motores, para el PWM y libera todos los pines.
    Llama a esta función SIEMPRE al salir del programa (bloque finally).
    """
    if not GPIO_DISPONIBLE:
        print("[GPIO] Simulación: limpiar_gpio() no hace nada.")
        return

    detener_motores()

    if _pwm_ena is not None:
        _pwm_ena.stop()
    if _pwm_enb is not None:
        _pwm_enb.stop()

    GPIO.cleanup()
    print("[GPIO] Pines liberados correctamente.")


# =============================================================================
# REFERENCIAS POR COLOR
# =============================================================================
# Referencias por color
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_PATHS = {
    "Verde": os.path.join(BASE_DIR, "ref_verde.jpg"),
}

# =============================================================================
# RANGOS HSV (ajustar según iluminación real)
# =============================================================================
HSV_RANGES = {
    "Verde": [
        (np.array([40,  70,  60]),  np.array([85,  255, 255]))
    ],
}

KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# =============================================================================
# VARIABLES GLOBALES DE TRACKING
# =============================================================================
tracked_objects  = {}
next_object_id   = 1
color_id_count   = {"Verde": 0}

# [GPIO] Ya no se necesitan last_uart_cmd / last_uart_time.
# El GPIO actúa inmediatamente; no hay protocolo serial que respetar.
# Si quieres "debounce" de comandos, usa la variable siguiente:
_ultimo_cmd_gpio    = None
_ultimo_cmd_tiempo  = 0.0
GPIO_CMD_DEBOUNCE   = 0.08   # segundos mínimos entre cambios de comando


# =============================================================================
# FUNCIONES DE FEATURES  (ORB – sin cambios)
# =============================================================================

def crear_detector_y_matcher():
    detector = cv2.ORB_create(
        nfeatures=500,
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=15
    )
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return detector, matcher


def extraer_features(detector, imagen_gray):
    kp, des = detector.detectAndCompute(imagen_gray, None)
    return kp, des

def obtener_metricas_match(matcher, kp_ref, des_ref, kp_roi, des_roi, ratio=0.75):
    metrics = {
        "good_matches": 0,
        "inliers":      0,
        "inlier_ratio": 0.0,
        "homography_ok": False
    }

    if des_ref is None or des_roi is None:
        return metrics
    if kp_ref is None or kp_roi is None:
        return metrics
    if len(des_ref) < 2 or len(des_roi) < 2:
        return metrics

    matches = matcher.knnMatch(des_ref, des_roi, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

    metrics["good_matches"] = len(good)

    if len(good) < MIN_HOMOGRAPHY_POINTS:
        return metrics

    try:
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_roi[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts, dst_pts,
            cv2.RANSAC,
            RANSAC_REPROJ_THRESHOLD
        )

        if H is not None and mask is not None:
            inliers = int(mask.ravel().sum())
            metrics["inliers"]       = inliers
            metrics["inlier_ratio"]  = inliers / float(len(good)) if len(good) > 0 else 0.0
            metrics["homography_ok"] = inliers >= MIN_HOMOGRAPHY_POINTS

    except cv2.error:
        pass

    return metrics


def clasificar_estado_metricas(inliers, inlier_ratio, good_matches):
    if good_matches == 0:
        return "SIN MATCH"
    if inliers >= 15 and inlier_ratio >= 0.50:
        return "ROBUSTO"
    if inliers >= 8 and inlier_ratio >= 0.35:
        return "ACEPTABLE"
    if inliers >= 4:
        return "DEBIL"
    return "SIN HOMOGRAFIA"


# =============================================================================
# REFERENCIAS (sin cambios)
# =============================================================================

def cargar_referencias(detector):
    referencias = {}

    for color, path in REF_PATHS.items():
        img = cv2.imread(path)

        if img is None:
            print(f"[ADVERTENCIA] No se pudo cargar: {path}")
            referencias[color] = {"img": None, "gray": None, "kp": None, "des": None}
            continue

        gray        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des     = extraer_features(detector, gray)

        referencias[color] = {"img": img, "gray": gray, "kp": kp, "des": des}

    return referencias


# =============================================================================
# DETECCIÓN POR HSV + VALIDACIÓN CON ORB (sin cambios)
# =============================================================================

def detectar_cubos(frame_bgr, detector, matcher, referencias):
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    detections = []

    for color, ranges in HSV_RANGES.items():
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for low, high in ranges:
            mask       = cv2.inRange(hsv, low, high)
            mask_total = cv2.bitwise_or(mask_total, mask)

        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN,  KERNEL)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, KERNEL)

        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA or area > MAX_AREA:
                continue

            peri  = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, EPSILON_FRAC * peri, True)

            x, y, w, h = cv2.boundingRect(approx)
            if h == 0:
                continue

            aspect = w / float(h)
            if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
                continue

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame_bgr.shape[1], x + w), min(frame_bgr.shape[0], y + h)

            roi_gray     = gray[y1:y2, x1:x2]
            kp_roi, des_roi = extraer_features(detector, roi_gray)

            metricas = obtener_metricas_match(
                matcher,
                referencias[color]["kp"],
                referencias[color]["des"],
                kp_roi, des_roi,
                ratio=RATIO_TEST
            )

            estado_match = clasificar_estado_metricas(
                metricas["inliers"],
                metricas["inlier_ratio"],
                metricas["good_matches"]
            )

            cx = x + w // 2
            cy = y + h // 2

            detections.append({
                "color":        color,
                "bbox":         (x, y, w, h),
                "center":       (cx, cy),
                "matches":      metricas["good_matches"],
                "inliers":      metricas["inliers"],
                "inlier_ratio": metricas["inlier_ratio"],
                "status":       estado_match,
                "contour":      approx
            })

    return detections


# =============================================================================
# TRACKING (sin cambios)
# =============================================================================

def generar_label(color):
    global color_id_count
    color_id_count[color] += 1
    if color == "Verde": return f"G{color_id_count[color]}"
    return f"OBJ{color_id_count[color]}"


def registrar_objeto(detection):
    global tracked_objects, next_object_id
    obj_id             = next_object_id
    next_object_id    += 1
    tracked_objects[obj_id] = {
        "label":        generar_label(detection["color"]),
        "color":        detection["color"],
        "bbox":         detection["bbox"],
        "center":       detection["center"],
        "missed":       0,
        "matches":      detection["matches"],
        "inliers":      detection["inliers"],
        "inlier_ratio": detection["inlier_ratio"],
        "status":       detection["status"],
        "trail":        deque([detection["center"]], maxlen=TRAIL_LENGTH)
    }


def eliminar_objeto(obj_id):
    global tracked_objects
    if obj_id in tracked_objects:
        del tracked_objects[obj_id]


def actualizar_tracking(detections):
    global tracked_objects

    if len(detections) == 0:
        ids_borrar = []
        for obj_id in tracked_objects:
            tracked_objects[obj_id]["missed"] += 1
            if tracked_objects[obj_id]["missed"] > MAX_MISSED:
                ids_borrar.append(obj_id)
        for obj_id in ids_borrar:
            eliminar_objeto(obj_id)
        return tracked_objects

    if len(tracked_objects) == 0:
        for det in detections:
            registrar_objeto(det)
        return tracked_objects

    used_detections = set()
    object_ids      = list(tracked_objects.keys())

    for obj_id in object_ids:
        obj = tracked_objects[obj_id]
        best_det_idx, best_dist = -1, float("inf")

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
                best_dist    = d
                best_det_idx = i

        if best_det_idx != -1 and best_dist < MAX_DIST:
            det = detections[best_det_idx]
            tracked_objects[obj_id]["bbox"]         = det["bbox"]
            tracked_objects[obj_id]["center"]       = det["center"]
            tracked_objects[obj_id]["missed"]       = 0
            tracked_objects[obj_id]["matches"]      = det["matches"]
            tracked_objects[obj_id]["inliers"]      = det["inliers"]
            tracked_objects[obj_id]["inlier_ratio"] = det["inlier_ratio"]
            tracked_objects[obj_id]["status"]       = det["status"]
            tracked_objects[obj_id]["trail"].append(det["center"])
            used_detections.add(best_det_idx)
        else:
            tracked_objects[obj_id]["missed"] += 1

    ids_borrar = [oid for oid in tracked_objects if tracked_objects[oid]["missed"] > MAX_MISSED]
    for oid in ids_borrar:
        eliminar_objeto(oid)

    for i, det in enumerate(detections):
        if i not in used_detections:
            registrar_objeto(det)

    return tracked_objects


# =============================================================================
# OBJETIVO Y COMANDO DE CENTRADO (sin cambios en la lógica)
# =============================================================================

def seleccionar_objetivo(objetos, frame_shape):
    if len(objetos) == 0:
        return None

    h, w = frame_shape[:2]
    cx_frame, cy_frame = w // 2, h // 2

    candidatos = []
    for obj_id, obj in objetos.items():
        if obj["missed"] > 0:
            continue
        dx   = obj["center"][0] - cx_frame
        dy   = obj["center"][1] - cy_frame
        dist = hypot(dx, dy)
        candidatos.append((obj_id, obj["inliers"], obj["inlier_ratio"], obj["matches"], dist))

    if len(candidatos) == 0:
        return None

    candidatos.sort(key=lambda x: (-x[1], -x[2], -x[3], x[4]))
    return candidatos[0][0]


def calcular_comando_centrado(center_obj, frame_shape):
    """
    Regresa (instruccion_texto, cmd_gpio, dx, dy, centrado).
    El nombre interno cambió de 'uart_cmd' a 'cmd_gpio', pero los valores
    de los comandos (UP, DOWN, LEFT, etc.) son exactamente los mismos.
    """
    h, w = frame_shape[:2]
    cx_frame, cy_frame = w // 2, h // 2

    dx = center_obj[0] - cx_frame
    dy = center_obj[1] - cy_frame

    if dx < -CENTER_TOL_X:
        horiz, txt_h = "LEFT",  "IZQUIERDA"
    elif dx > CENTER_TOL_X:
        horiz, txt_h = "RIGHT", "DERECHA"
    else:
        horiz, txt_h = "", ""

    if dy < -CENTER_TOL_Y:
        vert, txt_v = "UP",   "ARRIBA"
    elif dy > CENTER_TOL_Y:
        vert, txt_v = "DOWN", "ABAJO"
    else:
        vert, txt_v = "", ""

    if horiz == "" and vert == "":
        return "CENTRADO", "CENTER", dx, dy, True

    if horiz != "" and vert != "":
        texto = f"MOVER {txt_h} | {txt_v}"
        cmd   = f"{horiz}_{vert}"
    elif horiz != "":
        texto = f"MOVER {txt_h}"
        cmd   = horiz
    else:
        texto = f"MOVER {txt_v}"
        cmd   = vert

    return texto, cmd, dx, dy, False


# =============================================================================
# [GPIO] WRAPPER CON DEBOUNCE  (reemplaza enviar_comando_uart)
# =============================================================================

def aplicar_comando_gpio(cmd: str):
    """
    [GPIO – reemplaza enviar_comando_uart()]
    Aplica el comando al L298N solo si cambió o pasó el tiempo de debounce.
    Esto evita escribir al GPIO en cada frame si el comando no cambió.
    """
    global _ultimo_cmd_gpio, _ultimo_cmd_tiempo

    now = time.time()

    if cmd != _ultimo_cmd_gpio or (now - _ultimo_cmd_tiempo) >= GPIO_CMD_DEBOUNCE:
        ejecutar_comando_movimiento(cmd)
        _ultimo_cmd_gpio   = cmd
        _ultimo_cmd_tiempo = now


# =============================================================================
# DIBUJO / HUD (mínimos cambios: "UART" → "GPIO")
# =============================================================================

def color_bgr(nombre):
    if nombre == "Verde": return (0, 255, 0)
    return (255, 255, 255)


def contar_por_color(objetos):
    conteo = {"Verde": 0}
    for _, obj in objetos.items():
        conteo[obj["color"]] += 1
    return conteo


def dibujar_centro(frame, objetivo_en_zona=False):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    color = (0, 255, 0) if objetivo_en_zona else (0, 255, 255)

    cv2.drawMarker(frame, (cx, cy), color,
                   markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2)
    cv2.rectangle(frame,
                  (cx - CENTER_TOL_X, cy - CENTER_TOL_Y),
                  (cx + CENTER_TOL_X, cy + CENTER_TOL_Y),
                  color, 2)
    cv2.putText(frame, "ZONA CENTRAL",
                (cx - 75, cy - CENTER_TOL_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def dibujar_hud(frame, objetivo, fps, avg_fps, elapsed_s, estado_hud, zona_hud):
    x0, y0 = 20, 210
    x1, y1 = 405, 430

    cv2.rectangle(frame, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(frame, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 1)

    matches     = objetivo["matches"]      if objetivo else 0
    inliers     = objetivo["inliers"]      if objetivo else 0
    inlier_ratio= objetivo["inlier_ratio"] if objetivo else 0.0
    etiqueta    = objetivo["label"]        if objetivo else "-"

    lineas = [
        f"Estado: {estado_hud}",
        f"Objetivo: {etiqueta}",
        f"Tiempo: {elapsed_s:6.1f}/{TEST_DURATION_SECONDS} s",
        f"FPS: {fps:.2f}",
        f"FPS promedio: {avg_fps:.2f}",
        f"Matches: {matches}",
        f"Inliers: {inliers}",
        f"Inlier-ratio: {inlier_ratio:.2f}",
        f"Zona central: {zona_hud}"
    ]

    y = y0 + 26
    for linea in lineas:
        cv2.putText(frame, linea, (x0 + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
        y += 22


def dibujar_resultados(frame, objetos, objetivo_id, instruccion,
                        cmd_gpio, dx, dy, fps, avg_fps, elapsed_s):
    """
    Igual que antes; 'uart_cmd' renombrado a 'cmd_gpio' en la firma.
    El texto en pantalla dice 'GPIO:' en lugar de 'UART:'.
    """
    h, w = frame.shape[:2]
    cx_frame, cy_frame = w // 2, h // 2

    objetivo         = objetos[objetivo_id] if objetivo_id is not None else None
    objetivo_en_zona = (objetivo is not None
                        and abs(dx) <= CENTER_TOL_X
                        and abs(dy) <= CENTER_TOL_Y)

    dibujar_centro(frame, objetivo_en_zona)

    for obj_id, obj in objetos.items():
        x, y, w_box, h_box = obj["bbox"]
        c      = color_bgr(obj["color"])
        grosor = 4 if obj_id == objetivo_id else 2

        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), c, grosor)

        texto = f'{obj["label"]} {obj["color"]} M:{obj["matches"]} I:{obj["inliers"]}'
        if obj_id == objetivo_id:
            texto += " [OBJETIVO]"

        cv2.putText(frame, texto, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)

        radio = 6 if obj_id == objetivo_id else 4
        cv2.circle(frame, obj["center"], radio, c, -1)

        pts = list(obj["trail"])
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], c, 2)

    if objetivo_id is not None:
        cv2.line(frame, (cx_frame, cy_frame), objetivo["center"], (0, 255, 255), 2)

        color_texto = (0, 255, 0) if cmd_gpio == "CENTER" else (0, 0, 255)

        cv2.putText(frame, f"Instruccion: {instruccion}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_texto, 2)

        # [GPIO] Texto actualizado: ya no dice "UART"
        cv2.putText(frame, f"GPIO: {cmd_gpio}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.putText(frame, f"Error X: {dx} px   Error Y: {dy} px", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if cmd_gpio == "CENTER":
            estado_hud = f"{objetivo['status']} | CENTRADO"
            zona_hud   = "DENTRO"
        else:
            estado_hud = f"{objetivo['status']} | AJUSTANDO"
            zona_hud   = "FUERA"

    else:
        cv2.putText(frame, "Instruccion: SIN OBJETIVO", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # [GPIO] Texto actualizado
        cv2.putText(frame, "GPIO: NO_TARGET", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        estado_hud = "SIN OBJETIVO"
        zona_hud   = "N/A"

    dibujar_hud(frame, objetivo, fps, avg_fps, elapsed_s, estado_hud, zona_hud)

    conteo = contar_por_color(objetos)
    total  = conteo["Verde"]

    cv2.putText(frame,
                f"#G: {conteo['Verde']}  "
                f"#Total: {total}",
                (20, 455),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.putText(frame, "Tecla: q=Salir", (20, 490),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)


# =============================================================================
# REGISTRO, CAPTURAS Y REPORTE (mínimo cambio: uart_cmd → cmd_gpio)
# =============================================================================

def preparar_directorios():
    os.makedirs(CAPTURE_DIR, exist_ok=True)


def crear_csv_session(path_csv):
    csv_file = open(path_csv, mode="w", newline="", encoding="utf-8")
    writer   = csv.writer(csv_file)
    writer.writerow([
        "frame_idx",
        "elapsed_s",
        "fps",
        "avg_fps",
        "num_objetos",
        "target_id",
        "target_label",
        "target_color",
        "target_status",
        "target_matches",
        "target_inliers",
        "target_inlier_ratio",
        "dx_px",
        "dy_px",
        "zona_central",
        "cmd_gpio",           # [GPIO] renombrado de uart_cmd → cmd_gpio
        "capture_path"
    ])
    return csv_file, writer


def generar_nombre_captura(frame_idx, elapsed_s, objetivo):
    etiqueta = objetivo.get("label", "OBJ") if objetivo is not None else "OBJ"
    color    = objetivo.get("color", "NA")  if objetivo is not None else "NA"
    return os.path.join(
        CAPTURE_DIR,
        f"cap_{frame_idx:06d}_{elapsed_s:06.2f}s_{etiqueta}_{color}.png".replace(" ", "_")
    )


def guardar_captura_automatica(frame, objetivo, frame_idx, elapsed_s):
    path = generar_nombre_captura(frame_idx, elapsed_s, objetivo)
    ok   = cv2.imwrite(path, frame)
    if ok:
        print(f"[CAPTURA] Guardada: {path}")
        return path
    print(f"[CAPTURA] No se pudo guardar: {path}")
    return ""


def registrar_frame_csv(writer, frame_idx, elapsed_s, fps, avg_fps,
                         objetos, objetivo_id, cmd_gpio, dx, dy,
                         objetivo_en_zona, capture_path):
    objetivo = (objetos[objetivo_id]
                if objetivo_id is not None and objetivo_id in objetos
                else None)

    writer.writerow([
        frame_idx,
        f"{elapsed_s:.4f}",
        f"{fps:.4f}",
        f"{avg_fps:.4f}",
        len(objetos),
        objetivo_id if objetivo_id is not None else "",
        objetivo["label"]        if objetivo is not None else "",
        objetivo["color"]        if objetivo is not None else "",
        objetivo["status"]       if objetivo is not None else "SIN OBJETIVO",
        objetivo["matches"]      if objetivo is not None else 0,
        objetivo["inliers"]      if objetivo is not None else 0,
        f"{objetivo['inlier_ratio']:.4f}" if objetivo is not None else "0.0000",
        dx,
        dy,
        "DENTRO" if objetivo_en_zona else ("FUERA" if objetivo is not None else "N/A"),
        cmd_gpio,                # [GPIO] antes era uart_cmd
        capture_path
    ])


def generar_grafica_fps(times_s, fps_values, output_path):
    if plt is None:
        print("[REPORTE] matplotlib no instalado. Sin gráfica de FPS.")
        return False
    if not times_s or not fps_values:
        print("[REPORTE] Sin datos para graficar.")
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
    print(f"[REPORTE] Gráfica generada: {output_path}")
    return True


def guardar_resumen_sesion(path_summary, elapsed_s, total_frames, avg_fps, captures_count):
    cumple = avg_fps >= MIN_REQUIRED_FPS
    with open(path_summary, "w", encoding="utf-8") as f:
        f.write("=== RESUMEN DE SESION ===\n")
        f.write(f"Duracion registrada: {elapsed_s:.2f} s\n")
        f.write(f"Frames procesados: {total_frames}\n")
        f.write(f"FPS promedio: {avg_fps:.2f}\n")
        f.write(f"Capturas automaticas: {captures_count}\n")
        f.write(f"Resolucion configurada: {FRAME_WIDTH}x{FRAME_HEIGHT}\n")
        f.write(f"Control: GPIO directo (sin UART/Arduino)\n")   # [GPIO]
        f.write(f"Meta FPS >= {MIN_REQUIRED_FPS:.0f}: {'CUMPLE' if cumple else 'NO CUMPLE'}\n")
        if not cumple:
            f.write("Nota: si no se alcanzan 15 FPS, documenta la justificacion tecnica.\n")
    print(f"[REPORTE] Resumen guardado: {path_summary}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # [GPIO] Inicializar pines ANTES de abrir la cámara
    inicializar_gpio()

    detector, matcher = crear_detector_y_matcher()
    referencias       = cargar_referencias(detector)

    # [GPIO] Ya NO se llama a iniciar_uart() ni se crea objeto 'ser'

    preparar_directorios()
    csv_file, csv_writer = crear_csv_session(SESSION_CSV_PATH)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        csv_file.close()
        limpiar_gpio()   # [GPIO] limpiar antes de lanzar excepción
        raise RuntimeError("No se pudo abrir la cámara")

    prev_time          = time.time()
    session_start      = prev_time
    frame_idx          = 0
    fps_hist           = []
    time_hist          = []
    prev_target_in_zone = False
    last_capture_time  = 0.0
    captures_count     = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx  += 1
            detections  = detectar_cubos(frame, detector, matcher, referencias)
            objetos     = actualizar_tracking(detections)
            objetivo_id = seleccionar_objetivo(objetos, frame.shape)

            if objetivo_id is not None:
                objetivo = objetos[objetivo_id]
                instruccion, cmd_gpio, dx, dy, centrado = calcular_comando_centrado(
                    objetivo["center"], frame.shape
                )
            else:
                instruccion = "SIN OBJETIVO"
                cmd_gpio    = "NO_TARGET"
                dx, dy      = 0, 0
                centrado    = False

            # [GPIO] Enviar comando al L298N (reemplaza enviar_comando_uart)
            aplicar_comando_gpio(cmd_gpio)

            curr_time = time.time()
            delta     = curr_time - prev_time
            fps       = 1.0 / delta if delta > 0 else 0.0
            prev_time = curr_time

            elapsed_s = curr_time - session_start
            fps_hist.append(fps)
            time_hist.append(elapsed_s)
            avg_fps = sum(fps_hist) / len(fps_hist)

            dibujar_resultados(frame, objetos, objetivo_id,
                               instruccion, cmd_gpio, dx, dy,
                               fps, avg_fps, elapsed_s)

            objetivo_en_zona = objetivo_id is not None and centrado
            capture_path     = ""

            if (objetivo_en_zona
                    and not prev_target_in_zone
                    and (curr_time - last_capture_time) >= CAPTURE_COOLDOWN_SECONDS):
                objetivo_obj = objetos.get(objetivo_id)
                capture_path = guardar_captura_automatica(frame, objetivo_obj, frame_idx, elapsed_s)
                if capture_path:
                    captures_count   += 1
                    last_capture_time = curr_time

            prev_target_in_zone = objetivo_en_zona

            registrar_frame_csv(
                csv_writer, frame_idx, elapsed_s, fps, avg_fps,
                objetos, objetivo_id, cmd_gpio, dx, dy,
                objetivo_en_zona, capture_path
            )

            # [GPIO] Título de ventana actualizado
            cv2.imshow("Tracking ORB + GPIO L298N", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if AUTO_STOP_AFTER_TEST and elapsed_s >= TEST_DURATION_SECONDS:
                print(f"[SESION] {TEST_DURATION_SECONDS} segundos alcanzados. Fin de prueba.")
                break

    finally:
        # ── Limpieza garantizada ────────────────────────────────────────────
        final_elapsed = time.time() - session_start
        avg_fps       = (sum(fps_hist) / len(fps_hist)) if fps_hist else 0.0

        csv_file.flush()
        csv_file.close()

        generar_grafica_fps(time_hist, fps_hist, FPS_PLOT_PATH)
        guardar_resumen_sesion(SESSION_SUMMARY_PATH, final_elapsed,
                               frame_idx, avg_fps, captures_count)

        print(f"[CSV] Guardado en: {SESSION_CSV_PATH}  ({frame_idx} frames)")
        print(f"[CSV] FPS promedio: {avg_fps:.2f}")

        if avg_fps < MIN_REQUIRED_FPS:
            print("[ADVERTENCIA] FPS < 15. Justifica técnicamente en el reporte.")

        cap.release()
        cv2.destroyAllWindows()

        # [GPIO] Limpiar pines SIEMPRE, incluso si hubo excepción
        limpiar_gpio()


if __name__ == "__main__":
    main()


# =============================================================================
# NOTAS DE CONEXIÓN Y PRECAUCIONES ELÉCTRICAS
# =============================================================================
#
#  ESQUEMA DE CONEXIÓN  Raspberry Pi  →  L298N  →  Motores DC
#  ─────────────────────────────────────────────────────────────
#
#  Raspberry Pi (BCM)   L298N         Motor
#  ──────────────────   ──────────    ──────────────────────────
#  GPIO 12  (Pin 32)  → ENA          (enable motor izquierdo)
#  GPIO 17  (Pin 11)  → IN1          \
#  GPIO 27  (Pin 13)  → IN2          /  → OUT1, OUT2 → Motor Izquierdo
#
#  GPIO 13  (Pin 33)  → ENB          (enable motor derecho)
#  GPIO 23  (Pin 16)  → IN3          \
#  GPIO 24  (Pin 18)  → IN4          /  → OUT3, OUT4 → Motor Derecho
#
#  GND (Pin 6 o 9)    → GND L298N   (tierra común OBLIGATORIA)
#
#  Fuente externa 6-12 V DC          → VCC (terminal de potencia L298N)
#  NUNCA alimentes los motores con el pin 5V del Raspberry Pi.
#
#  PRECAUCIONES IMPORTANTES
#  ────────────────────────
#  1. TIERRA COMÚN: conecta el GND del Raspberry Pi al GND del L298N.
#     Sin este puente, los niveles lógicos serán erráticos o el circuito
#     no funcionará.
#
#  2. FUENTE SEPARADA: los motores DC de un carro típico consumen
#     300-800 mA cada uno. El Raspberry Pi NO puede suministrar eso por
#     sus pines GPIO (máx. 16 mA por pin, 50 mA total). Usa baterías o
#     una fuente DC separada conectada al terminal de potencia del L298N.
#
#  3. NIVEL LÓGICO: el RPi opera a 3.3 V; el L298N acepta de 2.3 V en
#     adelante → son compatibles directamente, sin divisor de tensión.
#
#  4. DISIPADOR TÉRMICO: el CI del L298N se calienta bajo carga.
#     Coloca el disipador incluido. Sin él, puede entrar en protección
#     térmica y los motores se detendrán solos.
#
#  5. DIODOS DE PROTECCIÓN: el L298N tiene diodos internos de flyback,
#     pero si usas motores de más de 2 A agrega diodos externos 1N4007
#     en paralelo con cada terminal de motor (cátodo al lado positivo).
#
#  6. GPIO.cleanup() en limpiar_gpio() libera los pines correctamente
#     y evita el warning "channel already in use" en ejecuciones
#     sucesivas. Siempre se llama desde el bloque finally del main().
#
#  TABLA DE MOVIMIENTOS
#  ────────────────────
#  Comando    │ Motor Izquierdo │ Motor Derecho │ Resultado
#  ───────────┼─────────────────┼───────────────┼────────────────────
#  UP         │ FWD  70 %       │ FWD  70 %     │ Avanzar recto
#  DOWN       │ REV  70 %       │ REV  70 %     │ Retroceder recto
#  LEFT       │ REV  70 %       │ FWD  70 %     │ Giro en sitio izq.
#  RIGHT      │ FWD  70 %       │ REV  70 %     │ Giro en sitio der.
#  LEFT_UP    │ FWD  35 %       │ FWD  70 %     │ Arco adelante-izq.
#  RIGHT_UP   │ FWD  70 %       │ FWD  35 %     │ Arco adelante-der.
#  LEFT_DOWN  │ REV  35 %       │ REV  70 %     │ Arco atrás-izq.
#  RIGHT_DOWN │ REV  70 %       │ REV  35 %     │ Arco atrás-der.
#  CENTER     │ OFF             │ OFF           │ Detenido (centrado)
#  NO_TARGET  │ OFF             │ OFF           │ Detenido (sin obj.)
# =============================================================================