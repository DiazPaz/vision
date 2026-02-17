import os
import cv2
import time
import numpy as np
import serial
from datetime import datetime

UART_port = "/dev/ttyUSB0"
BAUDRATE = 115200

CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480

MIN_AREA = 2500          # área mínima para aceptar detección
STABLE_FRAMES = 6        # frames seguidos para validar color
COOLDOWN_S = 1.5         # tiempo mínimo entre capturas

SAVE_DIR = "captures_test"
os.makedirs(SAVE_DIR, exist_ok=True)


RED1_LO = np.array([0,   120, 70])
RED1_HI = np.array([10,  255, 255])
RED2_LO = np.array([170, 120, 70])
RED2_HI = np.array([179, 255, 255])

GREEN_LO = np.array([35,  80,  60])
GREEN_HI = np.array([85,  255, 255])

BLUE_LO  = np.array([95,  80,  60])
BLUE_HI  = np.array([130, 255, 255])

WHITE_LO = np.array([0,   0,   200])
WHITE_HI = np.array([179, 60,  255])

#movimientos
MOVE_BY_COLOR = {
    "white": "AVANZAR (F)",
    "red":   "RETROCEDER (B)",
    "green": "GIRAR DERECHA (R)",
    "blue":  "GIRAR IZQUIERDA (L)",
    "none":  "DETENER (S)"
}

def open_serial(port):
    if not port:
        return None
    try:
        s = serial.Serial(port, BAUDRATE, timeout=0.05)
        print(f"UART abierto: {port}")
        return s
    except Exception as e:
        print(f"Error abriendo UART {port}: {e}")
        return None
    
port = open_serial(UART_port)

def now_stamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def largest_contour_bbox(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area <= 0:
        return None, 0
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, w, h), area

def detect_color(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, RED1_LO, RED1_HI),
        cv2.inRange(hsv, RED2_LO, RED2_HI)
    )
    green_mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    blue_mask  = cv2.inRange(hsv, BLUE_LO,  BLUE_HI)
    white_mask = cv2.inRange(hsv, WHITE_LO, WHITE_HI)

    red_mask   = clean_mask(red_mask)
    green_mask = clean_mask(green_mask)
    blue_mask  = clean_mask(blue_mask)
    white_mask = clean_mask(white_mask)

    candidates = []
    for name, m in [("red", red_mask), ("green", green_mask), ("blue", blue_mask), ("white", white_mask)]:
        bbox, area = largest_contour_bbox(m)
        if bbox is not None:
            candidates.append((name, bbox, area))

    if not candidates:
        return "none", None, 0

    name, bbox, area = max(candidates, key=lambda x: x[2])

    if area < MIN_AREA:
        return "none", None, area

    return name, bbox, area

def move_GPIOs(color):
    if not port:
        print("Port error!")
        return
    
    if color == "white":
        port.write(("WHITE\n").encode())   # ← Agregar \n
    elif color == "blue":
        port.write(("BLUE\n").encode())
    elif color == "red":
        port.write(("RED\n").encode())
    elif color == "green":
        port.write(("GREEN\n").encode())

def annotate(frame, color, bbox, area, stable_count):
    out = frame.copy()
    move = MOVE_BY_COLOR.get(color, "—")
    move_GPIOs(color)

    txt1 = f"Color: {color} | Area: {int(area)} | Stable: {stable_count}"
    txt2 = f"Movimiento: {move}"
    cv2.putText(out, txt1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(out, txt1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(out, txt2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(out, txt2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(out, color, (x, max(0, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

    return out

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("No pude abrir la cámara.")
        return

    last_color = "none"
    stable_count = 0
    last_save_t = 0.0

    print("Listo. Presiona 'q' para salir.")
    print("Tip: usa hojas o algo.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        color, bbox, area = detect_color(frame)

        # estabilidad anti-parpadeo
        if color != "none" and color == last_color:
            stable_count += 1
        elif color != last_color:
            stable_count = 1
            last_color = color
        else:
            # si es none, reinicia
            last_color = "none"
            stable_count = 0

        # "decisión" solo imprime cuando es estable
        if color != "none" and stable_count == STABLE_FRAMES:
            print(f"[DECISION] {color.upper()} -> {MOVE_BY_COLOR[color]}")

        # guardar evidencia
        t = time.time()
        if (color != "none") and (stable_count >= STABLE_FRAMES) and (t - last_save_t > COOLDOWN_S):
            filename = f"{color}_{now_stamp()}.jpg"
            path = os.path.join(SAVE_DIR, filename)

            annotated = annotate(frame, color, bbox, area, stable_count)
            cv2.imwrite(path, annotated)
            print(f"[CAPTURA] Guardada: {path}")
            last_save_t = t

        preview = annotate(frame, color, bbox, area, stable_count)
        cv2.imshow("Test Color Vision (PC)", preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
