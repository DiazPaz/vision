"""
=============================================================================
  VISIÓN COMPUTACIONAL + CONTROL GPIO  –  Raspberry Pi  (sin Arduino/UART)
  Proyecto Final  |  Segundo Entregable  |  Departamento de Ingeniería UDEM
=============================================================================

OPTIMIZACIONES APLICADAS (versión de alto rendimiento)
───────────────────────────────────────────────────────
  THREADING:
    • Hilo de captura  (CaptureThread): cap.read() en hilo dedicado con
      Queue(maxsize=2) — el loop principal nunca espera al sensor de cámara.
    • Hilo de CSV      (CSVWorker):     escrituras a disco en hilo separado;
      el loop encola la fila y continúa sin bloqueo.
    • Hilo de capturas (SaveWorker):    cv2.imwrite() asíncrono; ya no
      congela el pipeline cuando se guarda un PNG.

  VISIÓN:
    • Resolución de procesamiento: 640×360 (antes 1280×720).
      Las operaciones matriciales trabajan sobre 4× menos píxeles.
    • ORB: nfeatures 1500→400, nlevels 8→4.  Tiempo de detección ~3×
      más rápido con pérdida mínima de precisión a esta resolución.
    • detectar_cubos: solo se procesa el contorno de MAYOR área por color
      (antes todos los contornos válidos → hasta 12 corridas ORB/frame).
    • Conversiones de color: BGR→HSV y BGR→GRAY se hacen UNA sola vez
      por frame y se reutilizan en los tres colores.

  HUD / DIBUJO:
    • Eliminado frame.copy() + addWeighted() en cada frame (~2.7 MB
      de operaciones de mezcla evitadas).  El fondo del HUD ahora se
      dibuja con un rectángulo sólido semiopaco sobre ROI local.
    • avg_fps usa deque(maxlen=60) + suma incremental O(1) en lugar de
      sum() sobre la lista completa en Python puro cada frame.

  GPIO: sin cambios funcionales.
=============================================================================
"""

# ── Importaciones estándar ──────────────────────────────────────────────────
import csv
import os
import queue
import threading
import time
from collections import deque
from math import hypot

import cv2
import numpy as np

# ── [GPIO] Importar RPi.GPIO ────────────────────────────────────────────────
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
FRAME_WIDTH    = 640   # ← optimizado: 4× menos píxeles que 1280×720
FRAME_HEIGHT   = 360

RATIO_TEST     = 0.75

# Áreas escaladas a 640×360 (factor 0.25 respecto a 1280×720)
MIN_AREA       = 300
MAX_AREA       = 50_000
EPSILON_FRAC   = 0.03
ASPECT_MIN     = 0.75
ASPECT_MAX     = 1.30

MAX_DIST       = 40    # proporcional a la nueva resolución
MAX_MISSED     = 10
TRAIL_LENGTH   = 20

CENTER_TOL_X   = 25   # proporcional a la nueva resolución
CENTER_TOL_Y   = 25

MIN_HOMOGRAPHY_POINTS    = 4
RANSAC_REPROJ_THRESHOLD  = 5.0

# Sesión / prueba
TEST_DURATION_SECONDS = 90
AUTO_STOP_AFTER_TEST  = True
MIN_REQUIRED_FPS      = 15.0

# Capturas y archivos de sesión
CAPTURE_DIR              = "captures"
CAPTURE_COOLDOWN_SECONDS = 1.0
SESSION_CSV_PATH         = "session.csv"
FPS_PLOT_PATH            = "fps_vs_tiempo.png"
SESSION_SUMMARY_PATH     = "session_summary.txt"


# =============================================================================
# [GPIO] PINES DEL L298N
# =============================================================================
GPIO_IN1 = 17
GPIO_IN2 = 27
GPIO_ENA = 12

GPIO_IN3 = 23
GPIO_IN4 = 24
GPIO_ENB = 13

PWM_FREQ                  = 1000
VELOCIDAD_NORMAL          = 70
VELOCIDAD_DIAGONAL_RAPIDO = 70
VELOCIDAD_DIAGONAL_LENTO  = 35

_pwm_ena = None
_pwm_enb = None


# =============================================================================
# [GPIO] CONTROL DE MOTORES  (sin cambios)
# =============================================================================

def inicializar_gpio():
    global _pwm_ena, _pwm_enb
    if not GPIO_DISPONIBLE:
        print("[GPIO] Simulación: inicializar_gpio() no hace nada.")
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in (GPIO_IN1, GPIO_IN2, GPIO_IN3, GPIO_IN4):
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    GPIO.setup(GPIO_ENA, GPIO.OUT)
    GPIO.setup(GPIO_ENB, GPIO.OUT)
    _pwm_ena = GPIO.PWM(GPIO_ENA, PWM_FREQ)
    _pwm_enb = GPIO.PWM(GPIO_ENB, PWM_FREQ)
    _pwm_ena.start(0)
    _pwm_enb.start(0)
    print(f"[GPIO] Pines configurados. ENA={GPIO_ENA}, ENB={GPIO_ENB}")


def _set_motor_izquierdo(adelante: bool, velocidad: float):
    if not GPIO_DISPONIBLE:
        return
    GPIO.output(GPIO_IN1, GPIO.HIGH if adelante else GPIO.LOW)
    GPIO.output(GPIO_IN2, GPIO.LOW  if adelante else GPIO.HIGH)
    _pwm_ena.ChangeDutyCycle(velocidad)


def _set_motor_derecho(adelante: bool, velocidad: float):
    if not GPIO_DISPONIBLE:
        return
    GPIO.output(GPIO_IN3, GPIO.HIGH if adelante else GPIO.LOW)
    GPIO.output(GPIO_IN4, GPIO.LOW  if adelante else GPIO.HIGH)
    _pwm_enb.ChangeDutyCycle(velocidad)


def mover_adelante(velocidad=VELOCIDAD_NORMAL):
    if not GPIO_DISPONIBLE:
        return
    _set_motor_izquierdo(True, velocidad)
    _set_motor_derecho(True,   velocidad)


def mover_atras(velocidad=VELOCIDAD_NORMAL):
    if not GPIO_DISPONIBLE:
        return
    _set_motor_izquierdo(False, velocidad)
    _set_motor_derecho(False,   velocidad)


def girar_izquierda(velocidad=VELOCIDAD_NORMAL):
    if not GPIO_DISPONIBLE:
        return
    _set_motor_izquierdo(False, velocidad)
    _set_motor_derecho(True,    velocidad)


def girar_derecha(velocidad=VELOCIDAD_NORMAL):
    if not GPIO_DISPONIBLE:
        return
    _set_motor_izquierdo(True,  velocidad)
    _set_motor_derecho(False,   velocidad)


def arco_izquierda_adelante():
    if not GPIO_DISPONIBLE:
        return
    _set_motor_izquierdo(True, VELOCIDAD_DIAGONAL_LENTO)
    _set_motor_derecho(True,   VELOCIDAD_DIAGONAL_RAPIDO)


def arco_derecha_adelante():
    if not GPIO_DISPONIBLE:
        return
    _set_motor_izquierdo(True, VELOCIDAD_DIAGONAL_RAPIDO)
    _set_motor_derecho(True,   VELOCIDAD_DIAGONAL_LENTO)


def arco_izquierda_atras():
    if not GPIO_DISPONIBLE:
        return
    _set_motor_izquierdo(False, VELOCIDAD_DIAGONAL_LENTO)
    _set_motor_derecho(False,   VELOCIDAD_DIAGONAL_RAPIDO)


def arco_derecha_atras():
    if not GPIO_DISPONIBLE:
        return
    _set_motor_izquierdo(False, VELOCIDAD_DIAGONAL_RAPIDO)
    _set_motor_derecho(False,   VELOCIDAD_DIAGONAL_LENTO)


def detener_motores():
    if not GPIO_DISPONIBLE:
        return
    for pin in (GPIO_IN1, GPIO_IN2, GPIO_IN3, GPIO_IN4):
        GPIO.output(pin, GPIO.LOW)
    _pwm_ena.ChangeDutyCycle(0)
    _pwm_enb.ChangeDutyCycle(0)


def ejecutar_comando_movimiento(cmd: str):
    dispatch = {
        "UP":         mover_adelante,
        "DOWN":       mover_atras,
        "LEFT":       girar_izquierda,
        "RIGHT":      girar_derecha,
        "LEFT_UP":    arco_izquierda_adelante,
        "RIGHT_UP":   arco_derecha_adelante,
        "LEFT_DOWN":  arco_izquierda_atras,
        "RIGHT_DOWN": arco_derecha_atras,
    }
    fn = dispatch.get(cmd)
    if fn:
        fn()
    else:
        detener_motores()


def limpiar_gpio():
    if not GPIO_DISPONIBLE:
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
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REF_PATHS = {
    "Rojo":  os.path.join(BASE_DIR, "ref_rojo.jpg"),
    "Verde": os.path.join(BASE_DIR, "ref_verde.jpg"),
    "Azul":  os.path.join(BASE_DIR, "ref_azul.jpg"),
}

# =============================================================================
# RANGOS HSV
# =============================================================================
HSV_RANGES = {
    "Rojo": [
        (np.array([0,   120, 70]),  np.array([10,  255, 255])),
        (np.array([170, 120, 70]),  np.array([180, 255, 255])),
    ],
    "Verde": [
        (np.array([40, 70, 60]),  np.array([85, 255, 255])),
    ],
    "Azul": [
        (np.array([100, 120, 80]), np.array([125, 255, 255])),
    ],
}

# Kernel morfológico (3×3 es suficiente a 640×360; antes 5×5)
KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# =============================================================================
# VARIABLES GLOBALES DE TRACKING
# =============================================================================
tracked_objects = {}
next_object_id  = 1
color_id_count  = {"Rojo": 0, "Verde": 0, "Azul": 0}

_ultimo_cmd_gpio   = None
_ultimo_cmd_tiempo = 0.0
GPIO_CMD_DEBOUNCE  = 0.08


# =============================================================================
# THREADS DE SOPORTE
# =============================================================================

class CaptureThread(threading.Thread):
    """
    Hilo productor de frames.
    Lee cap.read() de forma continua y guarda el último frame disponible
    en una Queue(maxsize=2).  El loop principal siempre toma el más fresco
    sin bloquear la CPU esperando al sensor de cámara.
    """
    def __init__(self, cap):
        super().__init__(daemon=True, name="CaptureThread")
        self.cap    = cap
        self.queue  = queue.Queue(maxsize=2)
        self._stop  = threading.Event()

    def run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self.queue.put((False, None))
                break
            # Descarta el frame antiguo si la cola está llena (siempre fresco)
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put((True, frame))

    def read(self):
        """Bloquea hasta tener un frame nuevo (timeout 1 s)."""
        try:
            return self.queue.get(timeout=1.0)
        except queue.Empty:
            return False, None

    def stop(self):
        self._stop.set()


class CSVWorker(threading.Thread):
    """
    Hilo consumidor de filas CSV.
    El loop principal encola (writer, row) y este hilo escribe a disco
    sin bloquear el pipeline de visión.
    Sentinel None detiene el hilo.
    """
    def __init__(self):
        super().__init__(daemon=True, name="CSVWorker")
        self.queue = queue.Queue()

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            writer, row = item
            writer.writerow(row)
            self.queue.task_done()

    def enqueue(self, writer, row):
        self.queue.put((writer, row))

    def stop(self):
        self.queue.put(None)
        self.join(timeout=3)


class SaveWorker(threading.Thread):
    """
    Hilo para guardar capturas PNG de forma asíncrona.
    Encola (path, frame); el hilo llama a cv2.imwrite sin bloquear el loop.
    Sentinel None detiene el hilo.
    """
    def __init__(self):
        super().__init__(daemon=True, name="SaveWorker")
        self.queue  = queue.Queue()
        self._paths = []    # registro de rutas guardadas (thread-safe append)
        self._lock  = threading.Lock()

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            path, frame = item
            ok = cv2.imwrite(path, frame)
            if ok:
                with self._lock:
                    self._paths.append(path)
                print(f"[CAPTURA] Guardada: {path}")
            else:
                print(f"[CAPTURA] Error al guardar: {path}")
            self.queue.task_done()

    def enqueue(self, path, frame):
        # Se guarda una copia del frame para evitar que el buffer sea
        # modificado antes de que el hilo termine de escribir.
        self.queue.put((path, frame.copy()))

    def stop(self):
        self.queue.put(None)
        self.join(timeout=5)


# =============================================================================
# FEATURES ORB — parámetros reducidos para RPi 4
# =============================================================================

def crear_detector_y_matcher():
    """
    ORB optimizado para Raspberry Pi 4:
      nfeatures  1500 → 400  (principal ganancia de velocidad)
      nlevels    8    → 4    (menos escalas de pirámide)
      fastThreshold  15→ 20  (detector FAST más selectivo = menos kp a procesar)
    """
    detector = cv2.ORB_create(
        nfeatures=400,
        scaleFactor=1.2,
        nlevels=4,
        fastThreshold=20,
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
        "homography_ok": False,
    }
    if des_ref is None or des_roi is None:
        return metrics
    if kp_ref is None or kp_roi is None:
        return metrics
    if len(des_ref) < 2 or len(des_roi) < 2:
        return metrics

    matches = matcher.knnMatch(des_ref, des_roi, k=2)
    good = [m for pair in matches if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]

    metrics["good_matches"] = len(good)

    if len(good) < MIN_HOMOGRAPHY_POINTS:
        return metrics

    try:
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_roi[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
        if H is not None and mask is not None:
            inliers = int(mask.ravel().sum())
            metrics["inliers"]       = inliers
            metrics["inlier_ratio"]  = inliers / float(len(good))
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
# REFERENCIAS
# =============================================================================

def cargar_referencias(detector):
    referencias = {}
    for color, path in REF_PATHS.items():
        img = cv2.imread(path)
        if img is None:
            print(f"[ADVERTENCIA] No se pudo cargar: {path}")
            referencias[color] = {"img": None, "gray": None, "kp": None, "des": None}
            continue
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = extraer_features(detector, gray)
        referencias[color] = {"img": img, "gray": gray, "kp": kp, "des": des}
    return referencias


# =============================================================================
# DETECCIÓN HSV + ORB  — OPTIMIZADA
# =============================================================================

def detectar_cubos(frame_bgr, hsv, gray, detector, matcher, referencias):
    """
    Recibe hsv y gray ya precalculados fuera de la función para no repetir
    las conversiones de color por cada llamada (se hacen UNA vez en el loop).

    Cambio clave: solo se procesa ORB en el contorno de mayor área por color,
    no en todos.  Esto pasa de hasta 12 llamadas ORB/frame a máximo 3.
    """
    detections = []

    for color, ranges in HSV_RANGES.items():
        # ── Máscara HSV ────────────────────────────────────────────────────
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in ranges:
            mask_total = cv2.bitwise_or(mask_total, cv2.inRange(hsv, low, high))

        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN,  KERNEL)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, KERNEL)

        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # ── Filtrar por área y aspecto; guardar candidatos ─────────────────
        candidatos = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA or area > MAX_AREA:
                continue
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, EPSILON_FRAC * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            if h == 0:
                continue
            if not (ASPECT_MIN <= w / float(h) <= ASPECT_MAX):
                continue
            candidatos.append((area, cnt, approx, x, y, w, h))

        if not candidatos:
            continue

        # ── SOLO el contorno de mayor área recibe ORB + matching ───────────
        candidatos.sort(key=lambda t: t[0], reverse=True)
        _, cnt, approx, x, y, w, h = candidatos[0]

        x1 = max(0, x);           y1 = max(0, y)
        x2 = min(gray.shape[1], x + w);  y2 = min(gray.shape[0], y + h)

        roi_gray        = gray[y1:y2, x1:x2]
        kp_roi, des_roi = extraer_features(detector, roi_gray)

        metricas = obtener_metricas_match(
            matcher,
            referencias[color]["kp"],
            referencias[color]["des"],
            kp_roi, des_roi,
            ratio=RATIO_TEST,
        )

        estado_match = clasificar_estado_metricas(
            metricas["inliers"],
            metricas["inlier_ratio"],
            metricas["good_matches"],
        )

        detections.append({
            "color":        color,
            "bbox":         (x, y, w, h),
            "center":       (x + w // 2, y + h // 2),
            "matches":      metricas["good_matches"],
            "inliers":      metricas["inliers"],
            "inlier_ratio": metricas["inlier_ratio"],
            "status":       estado_match,
            "contour":      approx,
        })

    return detections


# =============================================================================
# TRACKING
# =============================================================================

def generar_label(color):
    global color_id_count
    color_id_count[color] += 1
    if color == "Rojo":  return f"R{color_id_count[color]}"
    if color == "Verde": return f"G{color_id_count[color]}"
    if color == "Azul":  return f"B{color_id_count[color]}"
    return f"OBJ{color_id_count[color]}"


def registrar_objeto(detection):
    global tracked_objects, next_object_id
    obj_id = next_object_id
    next_object_id += 1
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
        "trail":        deque([detection["center"]], maxlen=TRAIL_LENGTH),
    }


def eliminar_objeto(obj_id):
    tracked_objects.pop(obj_id, None)


def actualizar_tracking(detections):
    global tracked_objects

    if not detections:
        ids_borrar = [oid for oid, o in tracked_objects.items()
                      if o["missed"] + 1 > MAX_MISSED]
        for oid in ids_borrar:
            tracked_objects[oid]["missed"] += 1
            eliminar_objeto(oid)
        for oid in list(tracked_objects):
            if oid not in ids_borrar:
                tracked_objects[oid]["missed"] += 1
        return tracked_objects

    if not tracked_objects:
        for det in detections:
            registrar_objeto(det)
        return tracked_objects

    used = set()
    for obj_id in list(tracked_objects):
        obj = tracked_objects[obj_id]
        best_idx, best_dist = -1, float("inf")
        for i, det in enumerate(detections):
            if i in used or det["color"] != obj["color"]:
                continue
            d = hypot(det["center"][0] - obj["center"][0],
                      det["center"][1] - obj["center"][1])
            if d < best_dist:
                best_dist, best_idx = d, i
        if best_idx != -1 and best_dist < MAX_DIST:
            det = detections[best_idx]
            tracked_objects[obj_id].update({
                "bbox":         det["bbox"],
                "center":       det["center"],
                "missed":       0,
                "matches":      det["matches"],
                "inliers":      det["inliers"],
                "inlier_ratio": det["inlier_ratio"],
                "status":       det["status"],
            })
            tracked_objects[obj_id]["trail"].append(det["center"])
            used.add(best_idx)
        else:
            tracked_objects[obj_id]["missed"] += 1

    ids_borrar = [oid for oid, o in tracked_objects.items()
                  if o["missed"] > MAX_MISSED]
    for oid in ids_borrar:
        eliminar_objeto(oid)

    for i, det in enumerate(detections):
        if i not in used:
            registrar_objeto(det)

    return tracked_objects


# =============================================================================
# OBJETIVO Y COMANDO
# =============================================================================

def seleccionar_objetivo(objetos, frame_shape):
    if not objetos:
        return None
    h, w = frame_shape[:2]
    cx_f, cy_f = w // 2, h // 2
    candidatos = [
        (oid, o["inliers"], o["inlier_ratio"], o["matches"],
         hypot(o["center"][0] - cx_f, o["center"][1] - cy_f))
        for oid, o in objetos.items() if o["missed"] == 0
    ]
    if not candidatos:
        return None
    candidatos.sort(key=lambda x: (-x[1], -x[2], -x[3], x[4]))
    return candidatos[0][0]


def calcular_comando_centrado(center_obj, frame_shape):
    h, w   = frame_shape[:2]
    dx     = center_obj[0] - w // 2
    dy     = center_obj[1] - h // 2

    horiz  = "LEFT"  if dx < -CENTER_TOL_X else ("RIGHT" if dx > CENTER_TOL_X else "")
    vert   = "UP"    if dy < -CENTER_TOL_Y else ("DOWN"  if dy > CENTER_TOL_Y else "")
    txt_h  = "IZQUIERDA" if horiz == "LEFT" else ("DERECHA" if horiz == "RIGHT" else "")
    txt_v  = "ARRIBA"    if vert  == "UP"   else ("ABAJO"   if vert  == "DOWN"  else "")

    if not horiz and not vert:
        return "CENTRADO", "CENTER", dx, dy, True

    if horiz and vert:
        return f"MOVER {txt_h} | {txt_v}", f"{horiz}_{vert}", dx, dy, False
    if horiz:
        return f"MOVER {txt_h}", horiz, dx, dy, False
    return f"MOVER {txt_v}", vert, dx, dy, False


# =============================================================================
# [GPIO] WRAPPER CON DEBOUNCE
# =============================================================================

def aplicar_comando_gpio(cmd: str):
    global _ultimo_cmd_gpio, _ultimo_cmd_tiempo
    now = time.time()
    if cmd != _ultimo_cmd_gpio or (now - _ultimo_cmd_tiempo) >= GPIO_CMD_DEBOUNCE:
        ejecutar_comando_movimiento(cmd)
        _ultimo_cmd_gpio   = cmd
        _ultimo_cmd_tiempo = now


# =============================================================================
# DIBUJO / HUD — OPTIMIZADO
# =============================================================================

def color_bgr(nombre):
    return {"Rojo": (0, 0, 255), "Verde": (0, 255, 0), "Azul": (255, 0, 0)}.get(
        nombre, (255, 255, 255))


def contar_por_color(objetos):
    c = {"Rojo": 0, "Verde": 0, "Azul": 0}
    for o in objetos.values():
        c[o["color"]] += 1
    return c


def dibujar_centro(frame, objetivo_en_zona=False):
    h, w   = frame.shape[:2]
    cx, cy = w // 2, h // 2
    color  = (0, 255, 0) if objetivo_en_zona else (0, 255, 255)
    cv2.drawMarker(frame, (cx, cy), color,
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)
    cv2.rectangle(frame,
                  (cx - CENTER_TOL_X, cy - CENTER_TOL_Y),
                  (cx + CENTER_TOL_X, cy + CENTER_TOL_Y),
                  color, 1)
    cv2.putText(frame, "ZONA CENTRAL",
                (cx - 55, cy - CENTER_TOL_Y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def dibujar_hud(frame, objetivo, fps, avg_fps, elapsed_s, estado_hud, zona_hud):
    """
    HUD sin frame.copy() + addWeighted.
    Se dibuja un rectángulo sólido oscuro sobre el ROI, evitando la mezcla
    de dos buffers completos en cada frame.
    """
    x0, y0, x1, y1 = 10, 105, 260, 285
    cv2.rectangle(frame, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 1)

    matches      = objetivo["matches"]      if objetivo else 0
    inliers      = objetivo["inliers"]      if objetivo else 0
    inlier_ratio = objetivo["inlier_ratio"] if objetivo else 0.0
    etiqueta     = objetivo["label"]        if objetivo else "-"

    lineas = [
        f"Estado: {estado_hud}",
        f"Objetivo: {etiqueta}",
        f"T: {elapsed_s:5.1f}/{TEST_DURATION_SECONDS}s",
        f"FPS: {fps:.1f}  Prom: {avg_fps:.1f}",
        f"M:{matches} I:{inliers} R:{inlier_ratio:.2f}",
        f"Zona: {zona_hud}",
    ]
    y = y0 + 16
    for linea in lineas:
        cv2.putText(frame, linea, (x0 + 6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        y += 28


def dibujar_resultados(frame, objetos, objetivo_id,
                        instruccion, cmd_gpio, dx, dy,
                        fps, avg_fps, elapsed_s):
    h, w       = frame.shape[:2]
    cx_f, cy_f = w // 2, h // 2

    objetivo         = objetos.get(objetivo_id)
    objetivo_en_zona = (objetivo is not None
                        and abs(dx) <= CENTER_TOL_X
                        and abs(dy) <= CENTER_TOL_Y)

    dibujar_centro(frame, objetivo_en_zona)

    for obj_id, obj in objetos.items():
        x, y, bw, bh = obj["bbox"]
        c      = color_bgr(obj["color"])
        grosor = 3 if obj_id == objetivo_id else 1
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), c, grosor)
        texto = f'{obj["label"]} M:{obj["matches"]} I:{obj["inliers"]}'
        if obj_id == objetivo_id:
            texto += " [OBJ]"
        cv2.putText(frame, texto, (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1)
        cv2.circle(frame, obj["center"], 4 if obj_id == objetivo_id else 3, c, -1)
        pts = list(obj["trail"])
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], c, 1)

    if objetivo_id is not None:
        cv2.line(frame, (cx_f, cy_f), objetivo["center"], (0, 255, 255), 1)
        color_texto = (0, 255, 0) if cmd_gpio == "CENTER" else (0, 0, 255)
        cv2.putText(frame, f"Inst: {instruccion}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_texto, 1)
        cv2.putText(frame, f"GPIO: {cmd_gpio}", (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.putText(frame, f"dX:{dx:+d} dY:{dy:+d} px", (10, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1)
        estado_hud = f"{objetivo['status']} | {'CENTRADO' if cmd_gpio == 'CENTER' else 'AJUSTANDO'}"
        zona_hud   = "DENTRO" if cmd_gpio == "CENTER" else "FUERA"
    else:
        cv2.putText(frame, "Inst: SIN OBJETIVO", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(frame, "GPIO: NO_TARGET", (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        estado_hud = "SIN OBJETIVO"
        zona_hud   = "N/A"

    dibujar_hud(frame, objetivo, fps, avg_fps, elapsed_s, estado_hud, zona_hud)

    conteo = contar_por_color(objetos)
    total  = sum(conteo.values())
    cv2.putText(frame,
                f"#R:{conteo['Rojo']} #G:{conteo['Verde']} #B:{conteo['Azul']} T:{total}",
                (10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(frame, "q=Salir", (10, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)


# =============================================================================
# I/O: CSV, CAPTURAS, REPORTE
# =============================================================================

def preparar_directorios():
    os.makedirs(CAPTURE_DIR, exist_ok=True)


def crear_csv_session(path_csv):
    csv_file = open(path_csv, mode="w", newline="", encoding="utf-8")
    writer   = csv.writer(csv_file)
    writer.writerow([
        "frame_idx", "elapsed_s", "fps", "avg_fps", "num_objetos",
        "target_id", "target_label", "target_color", "target_status",
        "target_matches", "target_inliers", "target_inlier_ratio",
        "dx_px", "dy_px", "zona_central", "cmd_gpio", "capture_path",
    ])
    return csv_file, writer


def generar_nombre_captura(frame_idx, elapsed_s, objetivo):
    etiqueta = objetivo.get("label", "OBJ") if objetivo else "OBJ"
    color    = objetivo.get("color", "NA")  if objetivo else "NA"
    return os.path.join(
        CAPTURE_DIR,
        f"cap_{frame_idx:06d}_{elapsed_s:06.2f}s_{etiqueta}_{color}.png".replace(" ", "_"),
    )


def generar_grafica_fps(times_s, fps_values, output_path):
    if plt is None:
        return False
    if not times_s:
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
        f.write(f"Resolucion: {FRAME_WIDTH}x{FRAME_HEIGHT}\n")
        f.write(f"Control: GPIO directo (sin UART/Arduino)\n")
        f.write(f"Meta FPS >= {MIN_REQUIRED_FPS:.0f}: {'CUMPLE' if cumple else 'NO CUMPLE'}\n")
        if not cumple:
            f.write("Nota: si no se alcanzan 15 FPS, documenta la justificación técnica.\n")
    print(f"[REPORTE] Resumen guardado: {path_summary}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    inicializar_gpio()

    detector, matcher = crear_detector_y_matcher()
    referencias       = cargar_referencias(detector)
    preparar_directorios()

    # ── Abrir cámara ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    # Reducir buffer interno de OpenCV para evitar frames stale
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        limpiar_gpio()
        raise RuntimeError("No se pudo abrir la cámara")

    # ── Arrancar hilos de soporte ────────────────────────────────────────────
    capture_thread = CaptureThread(cap)
    csv_file, csv_writer = crear_csv_session(SESSION_CSV_PATH)
    csv_worker  = CSVWorker()
    save_worker = SaveWorker()

    capture_thread.start()
    csv_worker.start()
    save_worker.start()

    # ── Estado del loop ──────────────────────────────────────────────────────
    session_start       = time.time()
    prev_time           = session_start
    frame_idx           = 0
    fps_hist            = deque(maxlen=60)   # media móvil (60 frames)
    fps_sum             = 0.0                # suma incremental O(1)
    time_hist           = []
    prev_target_in_zone = False
    last_capture_time   = 0.0
    captures_count      = 0

    try:
        while True:
            # ── Obtener frame del hilo productor ────────────────────────────
            ret, frame = capture_thread.read()
            if not ret or frame is None:
                break

            # ── Conversiones de color: UNA sola vez por frame ───────────────
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── Procesamiento de visión ─────────────────────────────────────
            frame_idx  += 1
            detections  = detectar_cubos(frame, hsv, gray, detector, matcher, referencias)
            objetos     = actualizar_tracking(detections)
            objetivo_id = seleccionar_objetivo(objetos, frame.shape)

            if objetivo_id is not None:
                objetivo = objetos[objetivo_id]
                instruccion, cmd_gpio, dx, dy, centrado = calcular_comando_centrado(
                    objetivo["center"], frame.shape)
            else:
                instruccion = "SIN OBJETIVO"
                cmd_gpio    = "NO_TARGET"
                dx = dy     = 0
                centrado    = False

            aplicar_comando_gpio(cmd_gpio)

            # ── FPS con suma incremental ─────────────────────────────────────
            curr_time = time.time()
            delta     = curr_time - prev_time
            fps       = 1.0 / delta if delta > 0 else 0.0
            prev_time = curr_time

            elapsed_s = curr_time - session_start

            # Actualizar deque + suma sin recorrer toda la lista
            if len(fps_hist) == fps_hist.maxlen:
                fps_sum -= fps_hist[0]
            fps_hist.append(fps)
            fps_sum += fps
            avg_fps = fps_sum / len(fps_hist)

            time_hist.append(elapsed_s)

            # ── Dibujar ──────────────────────────────────────────────────────
            dibujar_resultados(frame, objetos, objetivo_id,
                               instruccion, cmd_gpio, dx, dy,
                               fps, avg_fps, elapsed_s)

            # ── Captura automática (async) ────────────────────────────────────
            objetivo_en_zona = objetivo_id is not None and centrado
            capture_path     = ""

            if (objetivo_en_zona
                    and not prev_target_in_zone
                    and (curr_time - last_capture_time) >= CAPTURE_COOLDOWN_SECONDS):
                objetivo_obj = objetos.get(objetivo_id)
                path = generar_nombre_captura(frame_idx, elapsed_s, objetivo_obj)
                save_worker.enqueue(path, frame)   # ← no bloquea
                capture_path      = path
                captures_count   += 1
                last_capture_time = curr_time

            prev_target_in_zone = objetivo_en_zona

            # ── CSV asíncrono ─────────────────────────────────────────────────
            objetivo_data = objetos.get(objetivo_id)
            csv_worker.enqueue(csv_writer, [
                frame_idx,
                f"{elapsed_s:.4f}",
                f"{fps:.4f}",
                f"{avg_fps:.4f}",
                len(objetos),
                objetivo_id if objetivo_id is not None else "",
                objetivo_data["label"]        if objetivo_data else "",
                objetivo_data["color"]        if objetivo_data else "",
                objetivo_data["status"]       if objetivo_data else "SIN OBJETIVO",
                objetivo_data["matches"]      if objetivo_data else 0,
                objetivo_data["inliers"]      if objetivo_data else 0,
                f"{objetivo_data['inlier_ratio']:.4f}" if objetivo_data else "0.0000",
                dx, dy,
                "DENTRO" if objetivo_en_zona else ("FUERA" if objetivo_data else "N/A"),
                cmd_gpio,
                capture_path,
            ])

            # ── Mostrar ───────────────────────────────────────────────────────
            cv2.imshow("Tracking ORB + GPIO L298N", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if AUTO_STOP_AFTER_TEST and elapsed_s >= TEST_DURATION_SECONDS:
                print(f"[SESION] {TEST_DURATION_SECONDS} s alcanzados. Fin.")
                break

    finally:
        # ── Detener hilos ────────────────────────────────────────────────────
        capture_thread.stop()
        csv_worker.stop()
        save_worker.stop()

        final_elapsed = time.time() - session_start
        avg_fps_final = fps_sum / len(fps_hist) if fps_hist else 0.0

        csv_file.flush()
        csv_file.close()

        fps_values = list(fps_hist)
        generar_grafica_fps(time_hist[-len(fps_values):], fps_values, FPS_PLOT_PATH)
        guardar_resumen_sesion(SESSION_SUMMARY_PATH, final_elapsed,
                               frame_idx, avg_fps_final, captures_count)

        print(f"[CSV] {frame_idx} frames  |  FPS promedio: {avg_fps_final:.2f}")

        if avg_fps_final < MIN_REQUIRED_FPS:
            print("[ADVERTENCIA] FPS < 15. Justifica técnicamente en el reporte.")

        cap.release()
        cv2.destroyAllWindows()
        limpiar_gpio()


if __name__ == "__main__":
    main()