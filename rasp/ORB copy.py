import cv2
import numpy as np
import os
import time

# ================= CONFIG =================
TEMPLATE_PATH = r"C:\Users\jesus\OneDrive\Documentos\UDEM\Programacion\C++\projects\helloworld\.vscode\VirtualEnv\Vision computacional\WhatsApp Image 2026-02-19 at 15.17.27.jpeg"

CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480

NFEATURES = 4000

RATIO = 0.75
RANSAC_THRESH = 5
MIN_GOOD = 10
MIN_INLIERS = 10

WINDOW_NAME = "ORB + Ratio + RANSAC (Green Box)"

# ===== Estabilidad para clasificar TP/FP =====
STABLE_FRAMES = 6         # cuantas detecciones seguidas para decir "TP"
IOU_THRESH = 0.25         # bbox debe parecerse al anterior para contar como estable
LED_HOLD_S = 0.35         # mantiene LEDs prendidos este tiempo despues de ultima TP

# ===== LEDs (Raspberry Pi GPIO, opcional) =====
LED_PINS = [17, 27, 22]   # BCM pins, cambialos si usas otros

def setup_leds():
    """
    Intenta configurar LEDs con gpiozero o RPi.GPIO.
    Si no estas en Raspberry Pi, regresa (None, None) y sigue sin LEDs.
    """
    try:
        from gpiozero import LED
        leds = [LED(p) for p in LED_PINS]
        for led in leds:
            led.off()
        print(f"LEDs listos con gpiozero (BCM): {LED_PINS}")
        return ("gpiozero", leds)
    except Exception:
        pass

    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        for p in LED_PINS:
            GPIO.setup(p, GPIO.OUT)
            GPIO.output(p, GPIO.LOW)
        print(f"LEDs listos con RPi.GPIO (BCM): {LED_PINS}")
        return ("rpigpio", GPIO)
    except Exception:
        print("Aviso: GPIO no disponible (PC o libreria faltante). LEDs desactivados.")
        return (None, None)

def set_leds(backend, obj, state: bool):
    if backend == "gpiozero":
        for led in obj:
            led.on() if state else led.off()
    elif backend == "rpigpio":
        GPIO = obj
        level = GPIO.HIGH if state else GPIO.LOW
        for p in LED_PINS:
            GPIO.output(p, level)

def cleanup_leds(backend, obj):
    try:
        if backend == "gpiozero":
            for led in obj:
                led.off()
                led.close()
        elif backend == "rpigpio":
            GPIO = obj
            for p in LED_PINS:
                GPIO.output(p, GPIO.LOW)
            GPIO.cleanup()
    except Exception:
        pass

def create_orb():
    return cv2.ORB_create(nfeatures=NFEATURES, fastThreshold=10)

def ratio_test(knn_matches, ratio):
    good = []
    for pair in knn_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good

def bbox_corners_from_H(H, w, h):
    if H is None:
        return None
    corners = np.float32([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]])
    try:
        pts = cv2.perspectiveTransform(corners, H)
        return pts
    except cv2.error:
        return None

def rect_from_corners(pts):
    p = pts.reshape(4, 2)
    x_min = int(np.min(p[:, 0]))
    y_min = int(np.min(p[:, 1]))
    x_max = int(np.max(p[:, 0]))
    y_max = int(np.max(p[:, 1]))
    return x_min, y_min, x_max, y_max

def clamp_rect(x1, y1, x2, y2, w, h, margin=0):
    x1 = max(-margin, min(w + margin, x1))
    y1 = max(-margin, min(h + margin, y1))
    x2 = max(-margin, min(w + margin, x2))
    y2 = max(-margin, min(h + margin, y2))
    return x1, y1, x2, y2

def is_rect_reasonable(x1, y1, x2, y2, frame_w, frame_h):
    if (x2 - x1) * (y2 - y1) < 1200:
        return False
    margin = 80
    if x2 < -margin or y2 < -margin or x1 > frame_w + margin or y1 > frame_h + margin:
        return False
    return True

def iou_rect(a, b):
    """Intersection over Union entre rects (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0

def main():
    # ===== LEDs init =====
    led_backend, led_obj = setup_leds()
    last_led_on_t = 0.0

    if not os.path.exists(TEMPLATE_PATH):
        print("Error: No existe el template:", TEMPLATE_PATH)
        cleanup_leds(led_backend, led_obj)
        return

    tpl = cv2.imread(TEMPLATE_PATH, 0)
    if tpl is None:
        print("Error: No se pudo leer el template:", TEMPLATE_PATH)
        cleanup_leds(led_backend, led_obj)
        return

    orb = create_orb()
    kp1, des1 = orb.detectAndCompute(tpl, None)

    if des1 is None or len(kp1) < 20:
        print("Error: Muy pocos keypoints en el template. Usa una imagen con mas textura/bordes.")
        cleanup_leds(led_backend, led_obj)
        return

    h1, w1 = tpl.shape[:2]
    print(f"Template listo | keypoints={len(kp1)} | size={w1}x{h1}")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: No se pudo abrir la camara")
        cleanup_leds(led_backend, led_obj)
        return

    # ===== Variables de estabilidad y conteo =====
    stable_count = 0
    last_bbox = None

    auto_tp = 0  # TP por regla de estabilidad
    auto_fp = 0  # FP por regla de inestabilidad
    manual_tp = 0
    manual_fp = 0
    last_auto_label = "NONE"

    last_t = time.perf_counter()
    fps = 0.0

    print("Teclas: q/ESC salir | T=manual TP | F=manual FP")
    print("Clasificacion auto: TP=estable | FP=inestable")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Error leyendo frame")
                break

            # FPS
            t = time.perf_counter()
            dt = t - last_t
            last_t = t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp2, des2 = orb.detectAndCompute(gray, None)

            frame_draw = frame.copy()
            status = "Buscando..."
            color = (0, 0, 255)
            good_n = 0
            inliers = 0

            candidate_detect = False
            bbox = None

            if des2 is not None and len(kp2) >= 20:
                knn = bf.knnMatch(des1, des2, k=2)
                good = ratio_test(knn, RATIO)
                good_n = len(good)

                if good_n >= MIN_GOOD:
                    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
                    inliers = int(mask.sum()) if mask is not None else 0

                    pts = bbox_corners_from_H(H, w1, h1)

                    if pts is not None and inliers >= MIN_INLIERS:
                        x1, y1, x2, y2 = rect_from_corners(pts)
                        x1, y1, x2, y2 = clamp_rect(x1, y1, x2, y2, FRAME_W, FRAME_H, margin=0)

                        if is_rect_reasonable(x1, y1, x2, y2, FRAME_W, FRAME_H):
                            candidate_detect = True
                            bbox = (x1, y1, x2, y2)
                        else:
                            status = "Homografia rara (rect invalido)"
                            color = (0, 0, 255)
                    else:
                        status = "Homografia debil"
                        color = (0, 0, 255)
                else:
                    status = "Pocos good matches"
                    color = (0, 0, 255)
            else:
                status = "Sin suficientes features"
                color = (0, 0, 255)

            # ===== Clasificacion TP/FP por estabilidad =====
            stable_tp = False
            if candidate_detect and bbox is not None:
                if last_bbox is None:
                    stable_count = 1
                else:
                    sim = iou_rect(bbox, last_bbox)
                    stable_count = (stable_count + 1) if sim >= IOU_THRESH else 1

                last_bbox = bbox

                if stable_count >= STABLE_FRAMES:
                    stable_tp = True
                    status = "TP (estable)"
                    color = (0, 255, 0)
                    # dibuja bbox verde
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
                else:
                    # hay deteccion, pero aun no es estable
                    status = "FP (inestable)"
                    color = (0, 255, 255)
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 255, 255), 2)
            else:
                # si no hay candidato, reinicia estabilidad
                stable_count = 0
                last_bbox = None

            # ===== Conteo automatico (solo cuando cambia etiqueta) =====
            # Nota: para no sumar por frame, sumamos cuando hay transicion a TP/FP
            if stable_tp and last_auto_label != "TP":
                auto_tp += 1
                last_auto_label = "TP"
            elif (candidate_detect and not stable_tp) and last_auto_label != "FP":
                auto_fp += 1
                last_auto_label = "FP"
            elif (not candidate_detect) and last_auto_label != "NONE":
                last_auto_label = "NONE"

            # ===== LEDs: prender solo en TP estable =====
            now = time.monotonic()
            if stable_tp:
                last_led_on_t = now
                set_leds(led_backend, led_obj, True)
            else:
                if (now - last_led_on_t) > LED_HOLD_S:
                    set_leds(led_backend, led_obj, False)

            # ===== HUD =====
            cv2.putText(
                frame_draw,
                f"{status} | good={good_n} | inliers={inliers} | stable={stable_count}/{STABLE_FRAMES} | FPS={fps:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )
            cv2.putText(
                frame_draw,
                f"AUTO TP={auto_tp} FP={auto_fp}  |  MANUAL TP={manual_tp} FP={manual_fp}  (T/F)",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(WINDOW_NAME, frame_draw)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:
                break
            elif key in [ord('t'), ord('T')]:
                manual_tp += 1
            elif key in [ord('f'), ord('F')]:
                manual_fp += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        cleanup_leds(led_backend, led_obj)
        print("Recursos liberados")

if __name__ == "__main__":
    main()
