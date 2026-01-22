# vision.py
import cv2 as cv
import numpy as np

# --- Rangos HSV ---
# Nota: estos rangos son "base" y pueden requerir ajuste según iluminación/cámara.
# Blanco se detecta mejor con S baja y V alta.
RANGES = {
    # Blanco (S baja, V alta)
    "Blanco":   (np.array([0,   0, 200]), np.array([180,  55, 255])),

    # Verde
    "Verde":    (np.array([40,  80,  60]), np.array([85, 255, 255])),

    # Azul
    "Azul":     (np.array([90, 120,  70]), np.array([130,255, 255])),

    # Rojo (dos bandas por el “wrap-around” del HSV)
    "Rojo1":    (np.array([0,  120, 120]), np.array([10, 255, 255])),
    "Rojo2":    (np.array([170,120, 120]), np.array([180,255, 255])),
}

# Colores para dibujar en BGR
DRAW = {
    "Blanco": (255, 255, 255),
    "Azul":   (255, 0, 0),
    "Verde":  (0, 255, 0),
    "Rojo":   (0, 0, 255),
}

MIN_AREA = 600
KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))


def process_mask(mask):
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, KERNEL, iterations=1)
    return mask


def find_and_draw(mask, frame_draw, label, draw=True):
    """
    - Detecta objetos del color (label)
    - Si draw=True, dibuja bounding boxes y centroides de los objetos en frame_draw
    - Devuelve:
        * found (bool)
        * detected_centroids (lista de tuplas: (label, cx, cy))
    """
    found = False
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    detected_centroids = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > MIN_AREA:
            found = True

            M = cv.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            detected_centroids.append((label, cx, cy))

            if draw:
                x, y, w, h = cv.boundingRect(cnt)

                cv.rectangle(frame_draw, (x, y), (x + w, y + h), DRAW[label], 2)
                cv.putText(frame_draw, label, (x, y - 8), cv.FONT_HERSHEY_SIMPLEX,
                           0.7, DRAW[label], 2, cv.LINE_AA)
                cv.circle(frame_draw, (cx, cy), 5, (255, 255, 255), -1)

    return found, detected_centroids


def detect_colors(frame, draw=True):
    """
    - Procesa frames y detecta: Blanco, Azul, Verde y Rojo
    - Devuelve:
        * detected_colors (set)
        * all_centroids (lista)
    - Sintaxis:
        * colors, centroids = detect_colors(frame)              # dibuja por default
        * colors, centroids = detect_colors(frame, draw=False)  # solo detecta
    """
    detected_colors = set()
    all_centroids = []

    blurred = cv.GaussianBlur(frame, (5, 5), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

    # --- Blanco ---
    low, high = RANGES["Blanco"]
    mask = process_mask(cv.inRange(hsv, low, high))
    found, centroids = find_and_draw(mask, frame, "Blanco", draw)
    if found:
        detected_colors.add("Blanco")
        all_centroids.extend(centroids)

    # --- Verde ---
    low, high = RANGES["Verde"]
    mask = process_mask(cv.inRange(hsv, low, high))
    found, centroids = find_and_draw(mask, frame, "Verde", draw)
    if found:
        detected_colors.add("Verde")
        all_centroids.extend(centroids)

    # --- Azul ---
    low, high = RANGES["Azul"]
    mask = process_mask(cv.inRange(hsv, low, high))
    found, centroids = find_and_draw(mask, frame, "Azul", draw)
    if found:
        detected_colors.add("Azul")
        all_centroids.extend(centroids)

    # --- Rojo (2 rangos) ---
    low1, high1 = RANGES["Rojo1"]
    low2, high2 = RANGES["Rojo2"]
    mask_red = cv.bitwise_or(cv.inRange(hsv, low1, high1), cv.inRange(hsv, low2, high2))
    mask_red = process_mask(mask_red)
    found, centroids = find_and_draw(mask_red, frame, "Rojo", draw)
    if found:
        detected_colors.add("Rojo")
        all_centroids.extend(centroids)

    return detected_colors, all_centroids
