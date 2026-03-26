"""
main.py — Punto de entrada del sistema de detección de piezas.

Modos:
  webcam   : python main.py --ref pieza.png
  video    : python main.py --ref pieza.png --video grabacion.mp4
  imagen   : python main.py --ref pieza.png --image fotograma.jpg
  galería  : python main.py --ref ref1.png ref2.png ref3.png

Controles interactivos (ventana de cámara/video):
  Q / ESC  → salir
  S        → guardar screenshot
  D        → toggle panel de debug
  K        → toggle keypoints
  P        → pausa / continua
  +/-      → ajustar umbral de detección en tiempo real
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from part_detector import PartDetector
import config


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_references(paths: list[str]) -> list[tuple[str, np.ndarray]]:
    refs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[main] ✗ No se pudo cargar: {p}")
            continue
        name = Path(p).stem
        refs.append((name, img))
        print(f"[main] ✓ Referencia cargada: '{name}'  ({img.shape[1]}×{img.shape[0]})")
    return refs


def overlay_hud(frame: np.ndarray, fps: float, threshold: float, paused: bool):
    h, w = frame.shape[:2]
    status = "PAUSADO" if paused else f"FPS {fps:.1f}"
    cv2.putText(frame, status, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(frame,
                f"Umbral: {threshold:.2f}  [+/-]",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def draw_color_rois(frame: np.ndarray):
    if not config.COLOR_RANGES:
        return
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fh, fw = frame.shape[:2]
    colors_draw = [(0, 165, 255), (255, 100, 0), (0, 255, 200)]
    for i, (lower, upper, name) in enumerate(config.COLOR_RANGES):
        m = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        col = colors_draw[i % len(colors_draw)]
        for c in cnts:
            if cv2.contourArea(c) >= config.COLOR_ROI_MIN_AREA:
                x, y, bw, bh = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), col, 1)
                cv2.putText(frame, name, (x+2, y-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)


# ──────────────────────────────────────────────────────────────────────────────
#  Modo imagen estática
# ──────────────────────────────────────────────────────────────────────────────

def run_image(detector: PartDetector, image_path: str):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[main] ✗ No se pudo abrir la imagen: {image_path}")
        return

    t0 = time.perf_counter()
    results = detector.detect(frame)
    elapsed = time.perf_counter() - t0

    PartDetector.draw(frame, results)
    out = PartDetector.draw_debug_panel(frame, results)

    print(f"\n── Resultados ({elapsed*1000:.1f} ms) ──")
    for r in results:
        if r.detected:
            print(f"  [{r.template_name}] score={r.score:.3f}  "
                  f"inliers={r.inliers}  contour={r.contour_match:.3f}  "
                  f"stable={r.stable}")
        else:
            print(f"  [{r.template_name}] no detectado")

    cv2.imshow("Detección — presiona Q para salir", out)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (ord('q'), 27):
            break
        if k == ord('s'):
            out_path = f"resultado_{int(time.time())}.png"
            cv2.imwrite(out_path, out)
            print(f"[main] Screenshot guardado: {out_path}")
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
#  Modo video / webcam
# ──────────────────────────────────────────────────────────────────────────────

def run_video(detector: PartDetector, source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[main] ✗ No se pudo abrir la fuente: {source}")
        return

    win = "Detector de piezas — [Q] salir  [S] screenshot  [D] debug  [K] kp  [P] pausa"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    show_debug   = True
    show_kp      = config.DRAW_KEYPOINTS
    paused       = False
    threshold    = config.DETECTION_THRESHOLD
    show_roi     = config.DRAW_ROI_COLOR

    fps_ring     = []
    frame_count  = 0

    print("[main] Captura iniciada. Controles: Q=salir  S=screenshot  D=debug  "
          "K=keypoints  P=pausa  +/-=umbral  R=ROIs")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str):   # video file — reiniciar
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

        t0 = time.perf_counter()

        if show_roi:
            draw_color_rois(frame)

        results = detector.detect(frame)

        elapsed = time.perf_counter() - t0
        fps_ring.append(1.0 / max(elapsed, 1e-6))
        if len(fps_ring) > 30:
            fps_ring.pop(0)
        fps = sum(fps_ring) / len(fps_ring)

        display = frame.copy()
        PartDetector.draw(display, results)
        overlay_hud(display, fps, threshold, paused)

        if show_debug:
            display = PartDetector.draw_debug_panel(display, results)

        cv2.imshow(win, display)
        frame_count += 1

        # ── Teclado ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            fn = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fn, display)
            print(f"[main] Screenshot: {fn}")
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('k'):
            show_kp = not show_kp
            config.DRAW_KEYPOINTS = show_kp
        elif key == ord('p'):
            paused = not paused
        elif key == ord('r'):
            show_roi = not show_roi
        elif key == ord('+'):
            config.DETECTION_THRESHOLD = min(threshold + 0.05, 0.95)
            threshold = config.DETECTION_THRESHOLD
            print(f"[main] Umbral → {threshold:.2f}")
        elif key == ord('-'):
            config.DETECTION_THRESHOLD = max(threshold - 0.05, 0.05)
            threshold = config.DETECTION_THRESHOLD
            print(f"[main] Umbral → {threshold:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[main] Procesados {frame_count} frames.")


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Detector de piezas industriales en tiempo real.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--ref", nargs="+", required=True,
        metavar="RUTA_IMAGEN",
        help="Una o más imágenes de referencia (recortadas alrededor de la pieza).",
    )
    p.add_argument(
        "--video", default=None,
        metavar="RUTA_VIDEO",
        help="Archivo de video. Si no se indica, se usa la webcam.",
    )
    p.add_argument(
        "--image", default=None,
        metavar="RUTA_IMAGEN",
        help="Procesa una imagen estática en vez de video/webcam.",
    )
    p.add_argument(
        "--camera", type=int, default=0,
        help="Índice de la cámara (default 0).",
    )
    p.add_argument(
        "--detector", choices=["SIFT", "AKAZE", "ORB"], default=None,
        help="Sobreescribe DETECTOR_TYPE de config.py.",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Sobreescribe DETECTION_THRESHOLD de config.py.",
    )
    p.add_argument(
        "--auto-mask", action="store_true",
        help="Genera máscara automática (fondo claro) para cada plantilla.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Sobreescrituras desde CLI
    if args.detector:
        config.DETECTOR_TYPE = args.detector
    if args.threshold is not None:
        config.DETECTION_THRESHOLD = args.threshold

    # Verificación de versión de OpenCV (necesitamos contrib para SIFT/AKAZE)
    print(f"[main] OpenCV {cv2.__version__}  |  "
          f"Detector: {config.DETECTOR_TYPE}  |  "
          f"Umbral: {config.DETECTION_THRESHOLD:.2f}")

    # Inicializar detector y cargar referencias
    detector = PartDetector()
    refs = load_references(args.ref)
    if not refs:
        print("[main] ✗ No se pudo cargar ninguna referencia. Saliendo.")
        sys.exit(1)

    for name, img in refs:
        detector.add_template(name, img, auto_mask=args.auto_mask)

    if not detector.templates:
        print("[main] ✗ Ninguna plantilla válida (keypoints insuficientes). "
              "Prueba con imágenes más grandes o con más textura.")
        sys.exit(1)

    # Ejecutar en el modo solicitado
    if args.image:
        run_image(detector, args.image)
    elif args.video:
        run_video(detector, args.video)
    else:
        run_video(detector, args.camera)


if __name__ == "__main__":
    main()
