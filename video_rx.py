from vision import detect_colors
import cv2 as cv
import requests
import time

PI_IP = "172.32.192.88"

VIDEO_URL  = f"http://{PI_IP}:5000/video_feed"
CMD_URL    = f"http://{PI_IP}:5000/command"
VISION_URL = f"http://{PI_IP}:5000/vision_data"

DETECTION_FRAMES = 15

# def send_rpms(left_rpm, right_rpm, dir_left, dir_right):   
#     speed = {
#         "left_rpm": f"S{left_rpm}",
#         "right_rpm": f"S{right_rpm}",


#         "left_dir": f"D{dir_left}",
#         "right_dir": f"D{dir_right}"
#     }

#     try:
#         requests.post(
#             CMD_URL,
#             json=speed,
#             timeout=0.2
#         )
#     except Exception as e:
#         print("Error enviando rpms:", e)


def send_vision_data(colors, centroids, areas):
    payload = {
        "colors": colors,
        "centroids": centroids,
        "areas": areas,
        "time": time.time()
    }

    try:
        requests.post(
            VISION_URL,
            json=payload,
            timeout=0.1
        )
    except Exception as e:
        print("Error enviando vision_data:", e)


# def calc_turn_x(centroid, frame_width, deadband_px=50):
#     cx = centroid[1]
#     center_x = frame_width / 2
#     error_px = cx - center_x

#     if abs(error_px) < deadband_px:
#         return 0.0

#     error = error_px / (frame_width / 2)   # [-1..1]
#     Kp = 0.8                                # ahora es “estable” por tamaño
#     turn = Kp * error

#     turn = max(-1.0, min(1.0, turn))       # turn normalizado
#     return turn


# def calc_base_y(centroid, frame_height, y_trigger_ratio=0.40):
#     cy = centroid[2]
#     y_trigger = frame_height * y_trigger_ratio

#     # Si el objeto aún está “arriba” (lejos), avanza
#     if cy < y_trigger:
#         return 1.0   # avanzar (normalizado)
#     else:
#         return 0.0   # ya llegó -> no avanzar


# def pick_target(centroids, areas, area_min=1500):
#     if not centroids or not areas:
#         return None  # no hay nada

#     # Filtra por área mínima
#     candidates = [(c, a) for c, a in zip(centroids, areas) if a >= area_min]
#     if not candidates:
#         return None

#     # Escoge el de mayor área
#     centroid, area = max(candidates, key=lambda x: x[1])
#     return centroid, area


def main():
    cap = cv.VideoCapture(VIDEO_URL)

    if not cap.isOpened():
        print("No se pudo abrir el stream")
        return

    print("Stream conectado")

    SEND_HZ = 15
    send_period = 1.0 / SEND_HZ
    last_send = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame no recibido")
            break

        # Detección
        colors, centroids, areas = detect_colors(frame, draw=True)

        # --- Envío de comandos ---
        now = time.time()
        if now - last_send >= send_period:
            # send_rpms(left_rpm, right_rpm, dir_left, dir_right)
            last_send = now

        # --- Cálculo de comandos ---
        send_vision_data(colors, centroids, areas)

        # Visualización
        cv.imshow("VISION (PC DEBUG)", frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
