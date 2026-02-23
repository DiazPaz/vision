import cv2
import numpy as np
import sys

# ============================================================
# Detección de bordes Canny en tiempo real (webcam)
# + Template Matching sobre el video feed
#
# USO:
#   python live_canny_template.py [template_image] [camera_index]
#   python live_canny_template.py template.png 0
#
# Ventanas:
#   - Live Feed       : cámara con rectángulo de match
#   - Canny Edges     : bordes del frame actual
#   - Template        : imagen de referencia
#   - Trackbars       : controles
# ============================================================

DEFAULT_TEMPLATE_PATH = r"rasp\WIN_20260212_15_18_24_Pro.png"
DEFAULT_CAMERA_INDEX  = 0

# ── parámetros iniciales ────────────────────────────────────
INIT_LOW        = 30    # 0..200
INIT_RATIO_X10  = 50    # 20..40  →  ratio = val/10  →  2.0..4.0
INIT_MATCH_THR  = 50    # 0..100  →  umbral = val/100 →  0.0..1.0

# ── globales ────────────────────────────────────────────────
template_gray   = None
template_color  = None
img_blur        = None
edges           = None

# ── match method ────────────────────────────────────────────
MATCH_METHOD = cv2.TM_CCOEFF_NORMED   # normalizado → 0..1

# ────────────────────────────────────────────────────────────
def canny_callback(_=0):
    """Recalcula Canny cuando cambia un trackbar."""
    global edges, img_blur
    if img_blur is None:
        return

    low       = max(0, cv2.getTrackbarPos('Low Threshold', 'Trackbars'))
    ratio     = max(20, cv2.getTrackbarPos('Ratio ×10',    'Trackbars')) / 10.0
    high      = int(low * ratio)

    edges = cv2.Canny(img_blur, low, high)
    cv2.imshow('Canny Edges', edges)


def clamp_ratio():
    """Fuerza el trackbar de ratio a ≥ 20 (2.0)."""
    val = cv2.getTrackbarPos('Ratio ×10', 'Trackbars')
    if val < 20:
        cv2.setTrackbarPos('Ratio ×10', 'Trackbars', 20)
        canny_callback()


def do_template_match(frame_gray, frame_color):
    """
    Busca template_gray dentro de frame_gray.
    Dibuja un rectángulo verde si la confianza supera el umbral.
    Devuelve el frame anotado y la confianza máxima.
    """
    if template_gray is None:
        return frame_color.copy(), 0.0

    th, tw = template_gray.shape[:2]
    fh, fw = frame_gray.shape[:2]

    # El template no puede ser más grande que el frame
    if th > fh or tw > fw:
        return frame_color.copy(), 0.0

    result    = cv2.matchTemplate(frame_gray, template_gray, MATCH_METHOD)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    match_thr = cv2.getTrackbarPos('Match Thr ×100', 'Trackbars') / 100.0

    annotated = frame_color.copy()
    if max_val >= match_thr:
        top_left     = max_loc
        bottom_right = (top_left[0] + tw, top_left[1] + th)
        cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 2)
        label = f"Match: {max_val:.2f}"
        cv2.putText(annotated, label, (top_left[0], max(top_left[1] - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(annotated, f"No match ({max_val:.2f})", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return annotated, max_val


# ────────────────────────────────────────────────────────────
def main():
    global template_gray, template_color, img_blur, edges

    # ── argumentos ──────────────────────────────────────────
    template_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEMPLATE_PATH
    cam_index     = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CAMERA_INDEX

    # ── cargar template ──────────────────────────────────────
    template_color = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template_color is None:
        print(f"[ERROR] No pude cargar el template: {template_path}")
        print("        Ejecuta: python live_canny_template.py <ruta_template> [cam_index]")
        sys.exit(1)

    template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)
    # Opcional: redimensionar template si es muy grande
    max_dim = 200
    th, tw  = template_gray.shape[:2]
    if max(th, tw) > max_dim:
        scale         = max_dim / max(th, tw)
        template_gray  = cv2.resize(template_gray,  (int(tw*scale), int(th*scale)))
        template_color = cv2.resize(template_color, (int(tw*scale), int(th*scale)))

    # ── abrir cámara ─────────────────────────────────────────
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] No pude abrir la cámara (índice {cam_index}).")
        sys.exit(1)

    # ── ventanas ─────────────────────────────────────────────
    for name in ('Live Feed', 'Canny Edges', 'Template', 'Trackbars'):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    cv2.imshow('Template', template_color)

    # ── trackbars ────────────────────────────────────────────
    cv2.createTrackbar('Low Threshold', 'Trackbars', INIT_LOW,       200, canny_callback)
    cv2.createTrackbar('Ratio ×10',     'Trackbars', INIT_RATIO_X10,  40, canny_callback)
    cv2.createTrackbar('Match Thr ×100','Trackbars', INIT_MATCH_THR, 100, lambda _: None)

    # forzar ratio mínimo = 20
    cv2.setTrackbarMin('Ratio ×10', 'Trackbars', 20)   # OpenCV 4.5+
    # fallback por si la versión no soporta setTrackbarMin
    clamp_ratio()

    print("[INFO] Presiona 'q' o ESC para salir.")
    print("[INFO] Ajusta los sliders para cambiar Canny y el umbral de match.")

    # ── loop principal ───────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame vacío, reintentando...")
            continue

        # Pre-proceso para Canny
        frame_gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_blur    = cv2.GaussianBlur(frame_gray, (5, 5), 1.4)

        # Canny
        low   = max(0, cv2.getTrackbarPos('Low Threshold', 'Trackbars'))
        ratio = max(20, cv2.getTrackbarPos('Ratio ×10',    'Trackbars')) / 10.0
        edges = cv2.Canny(img_blur, low, int(low * ratio))

        # Template matching
        annotated, confidence = do_template_match(frame_gray, frame)

        # Mostrar
        cv2.imshow('Live Feed',    annotated)
        cv2.imshow('Canny Edges',  edges)

        # Salida
        clamp_ratio()
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    # ── limpieza ─────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()