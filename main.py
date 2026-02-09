# main.py
import argparse
import sys
import time
import cv2 as cv

# Importa tu librería (asegúrate de que vision.py esté en el mismo folder que este main.py)
import vision


def open_capture(source: str, cam_index: int) -> cv.VideoCapture:
    """
    Abre una fuente de video:
    - Si source != "", se intenta abrir como archivo de video.
    - Si no, se abre la cámara cam_index.
    """
    if source:
        cap = cv.VideoCapture(source)
    else:
        cap = cv.VideoCapture(cam_index)

    return cap


def draw_centroid_list(frame, centroids):
    """
    Dibuja una lista de centroides (label, cx, cy) como texto en el frame.
    Nota: Los círculos + bounding boxes ya los dibuja vision.detect_colors(..., draw=True)
    """
    y0 = 25
    dy = 22

    if not centroids:
        cv.putText(frame, "Centroides: (ninguno)", (10, y0),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
        return

    cv.putText(frame, f"Centroides ({len(centroids)}):", (10, y0),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

    for i, (label, cx, cy) in enumerate(centroids[:10]):  # muestra hasta 10 para que no se sature
        text = f"- {label}: ({cx}, {cy})"
        cv.putText(frame, text, (10, y0 + (i + 1) * dy),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Deteccion de colores + contornos + centroides usando vision.py")
    parser.add_argument("--source", type=str, default="",
                        help="Ruta a un video (ej: videos/test.mp4). Si se deja vacio, usa webcam.")
    parser.add_argument("--cam", type=int, default=0,
                        help="Indice de camara (default 0). Solo aplica si --source esta vacio.")
    parser.add_argument("--no-draw", action="store_true",
                        help="Si se activa, NO dibuja bounding boxes/centroides (solo detecta).")
    parser.add_argument("--width", type=int, default=0, help="Ancho deseado (0 = no forzar).")
    parser.add_argument("--height", type=int, default=0, help="Alto deseado (0 = no forzar).")
    args = parser.parse_args()

    cap = open_capture(args.source, args.cam)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la fuente de video/camara.")
        print(" - Si usas webcam, prueba --cam 0, --cam 1, etc.")
        print(" - Si usas archivo, revisa la ruta en --source")
        sys.exit(1)

    # Opcional: forzar resolucion
    if args.width > 0:
        cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    draw = not args.no_draw

    last_time = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # Si es video y se acaba, salimos. Si es camara y falla, intentamos salir con mensaje.
            print("Fin del stream o no se pudo leer frame.")
            break

        # Tu libreria detecta y (opcionalmente) dibuja:
        detected_colors, centroids = vision.detect_colors(frame, draw=draw)

        # Overlay: colores detectados
        colors_text = "Colores: " + (", ".join(sorted(detected_colors)) if detected_colors else "(ninguno)")
        cv.putText(frame, colors_text, (10, frame.shape[0] - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

        # Overlay: centroides (lista)
        draw_centroid_list(frame, centroids)

        # Calculo simple de FPS (solo para referencia)
        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

        cv.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 45),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow("Deteccion (vision.py)", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            # Print “bonito” en consola de los centroides actuales
            print("Centroides detectados:", centroids)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

