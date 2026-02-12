import cv2
import numpy as np
import sys

# ============================================================
# Detección de bordes con Canny + Trackbars (OpenCV 4.x)
# - Carga imagen desde sys.argv[1] o usa una ruta por defecto
# - Muestra 4 ventanas: Original, Grayscale + Blur, Canny Edges, Trackbars
# - Ajuste en tiempo real de umbrales (low) y relación (ratio) para el high
# ============================================================

# --- Ruta por defecto (cámbiala a tu gusto) ---
DEFAULT_IMAGE_PATH = r"vision\rasp\Screenshot 2026-02-09 210243.png"
# Ejemplo Windows:
# DEFAULT_IMAGE_PATH = r"C:\fotos\prueba.jpg"

# Variables globales para compartir entre callback y loop
img_original = None
img_gray = None
img_blur = None
edges = None

# Parámetros iniciales (según requisito)
INIT_LOW = 50          # 0..200
INIT_RATIO_X10 = 30    # 20..40  -> ratio real = (valor/10.0) => 2.0..4.0

def canny_callback(_val=0):
    """
    Callback que se ejecuta cada vez que se mueve un trackbar.
    Lee los valores actuales de trackbars y actualiza la ventana de bordes.
    """
    global edges, img_blur

    # Leer valores de trackbars
    low = cv2.getTrackbarPos('Low Threshold', 'Trackbars')
    ratio_val = cv2.getTrackbarPos('Ratio ×10', 'Trackbars') / 10.0 # 40

    # Asegurar que low no sea negativo (por seguridad) y calcular high
    low = max(0, low) # 41
    print(low)

    high = int(low * ratio_val)
    print(high)

    # Ejecutar Canny sobre la imagen suavizada
    edges = cv2.Canny(img_blur, low, high)

    # Mostrar resultados
    cv2.imshow('Canny Edges', edges)

def main():
    global img_original, img_gray, img_blur, edges

    # ------------------------------------------------------------
    # 1) Obtener ruta de imagen desde argumentos o usar default
    # ------------------------------------------------------------
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = DEFAULT_IMAGE_PATH

    # ------------------------------------------------------------
    # 2) Cargar imagen (BGR). Manejar error si no carga.
    # ------------------------------------------------------------
    img_original = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_original is None:
        print(f"[ERROR] No pude cargar la imagen: {image_path}")
        print("        - Verifica la ruta y el nombre del archivo.")
        print("        - Si estás en VS Code / PyCharm, revisa el 'working directory'.")
        sys.exit(1)

    # ------------------------------------------------------------
    # 3) Pre-procesamiento: gris + GaussianBlur
    #    Kernel 5x5 y sigma=1.4 (como lo pides)
    # ------------------------------------------------------------
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1.4)

    # ------------------------------------------------------------
    # 4) Crear ventanas (4 ventanas requeridas)
    # ------------------------------------------------------------
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Grayscale + Blur', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Canny Edges', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)

    # Mostrar las 2 primeras de inmediato
    cv2.imshow('Original', img_original)
    cv2.imshow('Grayscale + Blur', img_blur)

    # ------------------------------------------------------------
    # 5) Crear trackbars en la ventana "Trackbars"
    #    lowThreshold: 0..200 (init 50)
    #    ratio: 20..40 (init 30) -> dividido entre 10 => 2.0..4.0
    # ------------------------------------------------------------
    cv2.createTrackbar('Low Threshold', 'Trackbars', INIT_LOW, 200, canny_callback)
    cv2.createTrackbar('Ratio ×10', 'Trackbars', INIT_RATIO_X10, 40, canny_callback)

    # Nota: el mínimo del trackbar no puede ser 20 directamente;
    # así que lo dejamos 0..40 y forzamos a 20..40 en el callback / loop.
    # Para cumplir estrictamente 20..40, ajustamos al vuelo:
    current_ratio_x10 = cv2.getTrackbarPos('Ratio ×10', 'Trackbars')
    if current_ratio_x10 < 20:
        cv2.setTrackbarPos('Ratio ×10', 'Trackbars', 20)

    # Calcular y mostrar bordes iniciales
    canny_callback(0)

    # ------------------------------------------------------------
    # 6) Loop para mantener ventanas activas y actualizar en tiempo real
    #    - cv2.waitKey(1) en while
    #    - Salir con 'q' o 'Esc'
    # ------------------------------------------------------------
    while True:
        # Mantener ratio dentro de 20..40 (2.0..4.0) como requisito
        ratio_x10 = cv2.getTrackbarPos('Ratio ×10', 'Trackbars')
        if ratio_x10 < 20:
            cv2.setTrackbarPos('Ratio ×10', 'Trackbars', 20)
            canny_callback(0)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC o 'q'
            break

        # (Opcional) Si quieres forzar refresco constante aunque no muevas sliders:
        # canny_callback(0)

    # ------------------------------------------------------------
    # 7) Liberar recursos
    # ------------------------------------------------------------
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()