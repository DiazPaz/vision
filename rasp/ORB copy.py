import cv2
import numpy as np
import os
import time

# ================= CONFIG =================
TEMPLATE_PATH = r"C:\Users\jesus\OneDrive\Documentos\UDEM\Programacion\C++\projects\helloworld\.vscode\VirtualEnv\Vision computacional\WhatsApp Image 2026-02-19 at 15.17.27.jpeg"

CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480

NFEATURES = 4000

# Más permisivo (para que vuelva a detectar)
RATIO = 0.75           # antes 0.75 (más alto = más matches)
RANSAC_THRESH = 5  #antes 5.0 (más alto = más tolerante)
MIN_GOOD = 10          # antes 25 (más bajo = detecta más)
MIN_INLIERS = 10       # antes 15-18 (más bajo = detecta más)

WINDOW_NAME = "ORB + Ratio + RANSAC (Green Box)"
# ==========================================

def create_orb():
    # fastThreshold bajo = más keypoints (mejor cuando hay poca textura/iluminación)
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
        pts = cv2.perspectiveTransform(corners, H)  # (4,1,2)
        return pts
    except cv2.error:
        return None

def rect_from_corners(pts):
    """Regresa (x1,y1,x2,y2) del rectángulo que encierra las 4 esquinas transformadas."""
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
    # área mínima para evitar rectángulos degenerados
    if (x2 - x1) * (y2 - y1) < 1200:
        return False
    # si está demasiado fuera, lo descartamos
    margin = 80
    if x2 < -margin or y2 < -margin or x1 > frame_w + margin or y1 > frame_h + margin:
        return False
    return True

def main():
    if not os.path.exists(TEMPLATE_PATH):
        print("Error: No existe el template:", TEMPLATE_PATH)
        return

    tpl = cv2.imread(TEMPLATE_PATH, 0)
    if tpl is None:
        print("Error: No se pudo leer el template:", TEMPLATE_PATH)
        return

    orb = create_orb()
    kp1, des1 = orb.detectAndCompute(tpl, None)

    if des1 is None or len(kp1) < 20:
        print("Error: Muy pocos keypoints en el template. Usa una imagen con más textura/bordes.")
        return

    h1, w1 = tpl.shape[:2]
    print(f"Template listo | keypoints={len(kp1)} | size={w1}x{h1}")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return

    last_t = time.perf_counter()
    fps = 0.0

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
                            # RECTÁNGULO VERDE cuando detecta
                            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            status = "OBJETO DETECTADO"
                            color = (0, 255, 0)
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

            cv2.putText(frame_draw,
                        f"{status} | good={good_n} | inliers={inliers} | FPS={fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2)

            cv2.imshow(WINDOW_NAME, frame_draw)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Recursos liberados")

if __name__ == "__main__":
    main()