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

MIN_AREA = 2500
STABLE_FRAMES = 6
COOLDOWN_S = 1.5

# NUEVO: limitar envio UART (no spamear)
UART_MIN_INTERVAL_S = 0.12   # 120 ms aprox (ajustable)

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

MOVE_BY_COLOR = {
    "white": "AVANZAR (F)",
    "red":   "RETROCEDER (B)",
    "green": "GIRAR DERECHA (R)",
    "blue":  "GIRAR IZQUIERDA (L)",
    "none":  "DETENER (S)"
}

# Mapeo directo a comandos UART para Arduino
CMD_BY_COLOR = {
    "white": "WHITE\n",
    "red":   "RED\n",
    "green": "GREEN\n",
    "blue":  "BLUE\n",
    "none":  "STOP\n"
}

def open_serial(port):
    if not port:
        return None
    try:
        s = serial.Serial(port, BAUDRATE, timeout=0.05)
<<<<<<< HEAD
        # bueno para evitar "pegados" al abrir
        s.reset_input_buffer()
        s.reset_output_buffer()
        print(f"UART abierto: {port} @ {BAUDRATE}")
=======
        time.sleep(2)  # Esperar reset de Arduino
        print(f"UART abierto: {port}")
>>>>>>> 223a30c98d1c34ad3aebc07f71489ad7636e4dd2
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

<<<<<<< HEAD
# NUEVO: envio UART controlado
_last_sent_cmd = None
_last_sent_t = 0.0

def send_uart_for_color(color):
    global _last_sent_cmd, _last_sent_t

    if not port:
        print("Port error!")
        return

    cmd = CMD_BY_COLOR.get(color, "STOP\n")
    t = time.time()

    # solo envia si cambio o si paso intervalo minimo
    if cmd != _last_sent_cmd or (t - _last_sent_t) > UART_MIN_INTERVAL_S:
        try:
            port.write(cmd.encode("ascii", errors="ignore"))
            _last_sent_cmd = cmd
            _last_sent_t = t
        except Exception as e:
            print(f"UART write error: {e}")
=======
# ✅ CORRECCIÓN 1: Agregar \n y manejar "none"
def move_GPIOs(color):
    if not port:
        print("Port error!")
        return
    
    try:
        if color == "white":
            port.write(b"WHITE\n")  # ← Usar bytes directamente con \n
        elif color == "blue":
            port.write(b"BLUE\n")
        elif color == "red":
            port.write(b"RED\n")
        elif color == "green":
            port.write(b"GREEN\n")
        elif color == "none":
            port.write(b"STOP\n")  # ← AGREGAR STOP para none
        
        port.flush()  # ← Asegurar envío inmediato
    except Exception as e:
        print(f"Error enviando comando: {e}")

>>>>>>> 223a30c98d1c34ad3aebc07f71489ad7636e4dd2

def annotate(frame, color, bbox, area, stable_count):
    out = frame.copy()
    move = MOVE_BY_COLOR.get(color, "—")
<<<<<<< HEAD
=======
    # ✅ NO llamar move_GPIOs aquí - se llama en main()
>>>>>>> 223a30c98d1c34ad3aebc07f71489ad7636e4dd2

    txt1 = f"Color: {color} | Area: {int(area)} | Stable: {stable_count}"
    txt2 = f"Movimiento: {move}"
    cv2.putText(out, txt1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(out, txt1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(out, txt2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(out, txt2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    if bbox is not None:
        x, y, w, h = bbox
        # si ya es estable, dibuja verde; si no, amarillo
        color_box = (0, 255, 0) if (color != "none" and stable_count >= STABLE_FRAMES) else (0, 255, 255)
        cv2.rectangle(out, (x, y), (x+w, y+h), color_box, 2)
        cv2.putText(out, color, (x, max(0, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_box, 2, cv2.LINE_AA)

    return out

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("No pude abrir la camara.")
        return

    last_color = "none"
    stable_count = 0
    last_save_t = 0.0
    last_sent_color = None  # ✅ CORRECCIÓN 2: Evitar envíos duplicados

    # NUEVO: para mandar STOP cuando se pierde deteccion estable
    last_decided_color = "none"

    print("Listo. Presiona 'q' para salir.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        color, bbox, area = detect_color(frame)

        # estabilidad anti-parpadeo (mas limpia)
        if color == "none":
            stable_count = 0
        else:
            if color == last_color:
                stable_count += 1
            else:
                last_color = color
                stable_count = 1

<<<<<<< HEAD
        # decision: solo cuando ya es estable
        decided_color = "none"
        if stable_count >= STABLE_FRAMES:
            decided_color = last_color
=======
        # ✅ CORRECCIÓN 3: Enviar comando solo cuando cambia y es estable
        if stable_count >= STABLE_FRAMES:
            if color != last_sent_color:
                move_GPIOs(color)
                last_sent_color = color
                print(f"[COMANDO] Enviado: {color.upper()} -> {MOVE_BY_COLOR[color]}")
        
        # Si perdemos estabilidad, enviar STOP
        elif stable_count < STABLE_FRAMES and last_sent_color != "none":
            move_GPIOs("none")
            last_sent_color = "none"
            print(f"[COMANDO] Enviado: STOP (perdida de estabilidad)")
>>>>>>> 223a30c98d1c34ad3aebc07f71489ad7636e4dd2

        # enviar UART SOLO si cambia la decision (o STOP si se perdio)
        if decided_color != last_decided_color:
            send_uart_for_color(decided_color)
            if decided_color != "none":
                print(f"[DECISION] {decided_color.upper()} -> {MOVE_BY_COLOR[decided_color]}")
            else:
                print("[DECISION] STOP")
            last_decided_color = decided_color

        # guardar evidencia (solo cuando esta estable)
        t = time.time()
        if (decided_color != "none") and (t - last_save_t > COOLDOWN_S):
            filename = f"{decided_color}_{now_stamp()}.jpg"
            path = os.path.join(SAVE_DIR, filename)

            annotated = annotate(frame, decided_color, bbox, area, stable_count)
            cv2.imwrite(path, annotated)
            print(f"[CAPTURA] Guardada: {path}")
            last_save_t = t

        preview = annotate(frame, decided_color, bbox, area, stable_count)
        cv2.imshow("Test Color Vision (PC)", preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ✅ CORRECCIÓN 4: Detener motores al salir
    if port:
        move_GPIOs("none")
        port.close()
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()