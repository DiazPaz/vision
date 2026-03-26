"""
gallery_builder.py — Genera una galería de plantillas rotadas/escaladas desde
una imagen de referencia, para mejorar la robustez con objetos vistos en
diferentes orientaciones (ver recomendación #1 del pipeline).

Uso:
    python gallery_builder.py pieza_original.png --angles 0 30 60 90 --output gallery/
    python gallery_builder.py pieza_original.png --auto --output gallery/
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  Utilidades de imagen
# ──────────────────────────────────────────────────────────────────────────────

def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rota la imagen manteniendo el contenido completo (amplía el canvas si
    es necesario para no recortar esquinas).
    """
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy
    return cv2.warpAffine(img, M, (new_w, new_h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def scale_image(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


def auto_crop(img: np.ndarray, pad: int = 10) -> np.ndarray:
    """Recorta el fondo blanco/oscuro alrededor de la pieza."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    all_pts = np.concatenate(cnts)
    x, y, bw, bh = cv2.boundingRect(all_pts)
    h, w = img.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)
    return img[y1:y2, x1:x2]


def add_background(img: np.ndarray, color=(240, 240, 240)) -> np.ndarray:
    """Reemplaza píxeles transparentes (si BGRA) o muy claros por un fondo sólido."""
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:   # BGRA
        mask = img[:, :, 3] == 0
        out  = img[:, :, :3].copy()
        out[mask] = color
    else:
        out = img.copy()
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Generador de galería
# ──────────────────────────────────────────────────────────────────────────────

def build_gallery(
    source_path: str,
    output_dir:  str,
    angles:      list[float],
    scales:      list[float],
    auto_crop_flag: bool = True,
) -> list[str]:
    """
    Genera archivos PNG para cada combinación (ángulo, escala).
    Devuelve la lista de rutas generadas.
    """
    src = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
    if src is None:
        raise FileNotFoundError(f"No se pudo cargar: {source_path}")
    src = add_background(src)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(source_path).stem
    generated = []

    for angle in angles:
        for scale in scales:
            variant = rotate_image(src, angle)
            if scale != 1.0:
                variant = scale_image(variant, scale)
            if auto_crop_flag:
                variant = auto_crop(variant)

            tag = f"a{int(angle):03d}_s{scale:.2f}".replace(".", "p")
            out_path = out_dir / f"{stem}_{tag}.png"
            cv2.imwrite(str(out_path), variant)
            generated.append(str(out_path))

    print(f"[gallery] Generadas {len(generated)} plantillas en: {out_dir}")
    return generated


def build_auto_gallery(
    source_path: str,
    output_dir:  str,
    n_angles:    int = 8,
    scales:      list[float] | None = None,
) -> list[str]:
    """
    Modo automático: genera n_angles rotaciones uniformes + escalas opcionales.
    """
    angles = [360 * i / n_angles for i in range(n_angles)]
    sc     = scales if scales else [1.0]
    return build_gallery(source_path, output_dir, angles, sc)


# ──────────────────────────────────────────────────────────────────────────────
#  Preview
# ──────────────────────────────────────────────────────────────────────────────

def preview_gallery(paths: list[str], max_cols: int = 4, thumb_size: int = 160):
    """Muestra un mosaico de todas las plantillas generadas."""
    thumbs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        ratio = min(thumb_size / w, thumb_size / h)
        th = cv2.resize(img, (int(w * ratio), int(h * ratio)))
        # canvas fijo
        canvas = np.full((thumb_size, thumb_size, 3), 50, dtype=np.uint8)
        oh = (thumb_size - th.shape[0]) // 2
        ow = (thumb_size - th.shape[1]) // 2
        canvas[oh:oh+th.shape[0], ow:ow+th.shape[1]] = th
        # etiqueta
        lbl = Path(p).stem[-12:]
        cv2.putText(canvas, lbl, (2, thumb_size - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)
        thumbs.append(canvas)

    if not thumbs:
        return

    n = len(thumbs)
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)
    mosaic = np.full((rows * thumb_size, cols * thumb_size, 3), 30, dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        mosaic[r*thumb_size:(r+1)*thumb_size, c*thumb_size:(c+1)*thumb_size] = t

    cv2.imshow("Galería de plantillas — Q para salir", mosaic)
    while True:
        if cv2.waitKey(0) & 0xFF in (ord('q'), 27):
            break
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Genera galería de plantillas rotadas/escaladas.",
        epilog=__doc__,
    )
    p.add_argument("source", help="Imagen de referencia original.")
    p.add_argument("--output", default="gallery",
                   help="Directorio de salida (default: gallery/).")
    p.add_argument("--angles", nargs="+", type=float, default=None,
                   help="Ángulos explícitos en grados. Ej: --angles 0 45 90 135")
    p.add_argument("--scales", nargs="+", type=float, default=[1.0],
                   help="Factores de escala. Ej: --scales 0.8 1.0 1.2")
    p.add_argument("--auto", action="store_true",
                   help="Modo automático: n rotaciones uniformes.")
    p.add_argument("--n-angles", type=int, default=8,
                   help="Número de rotaciones en modo --auto (default 8).")
    p.add_argument("--no-crop", action="store_true",
                   help="No recortar automáticamente cada variante.")
    p.add_argument("--preview", action="store_true",
                   help="Mostrar mosaico de preview al terminar.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.auto:
        paths = build_auto_gallery(
            args.source,
            args.output,
            n_angles=args.n_angles,
            scales=args.scales,
        )
    else:
        angles = args.angles if args.angles else [0, 45, 90, 135, 180, 225, 270, 315]
        paths  = build_gallery(
            args.source,
            args.output,
            angles=angles,
            scales=args.scales,
            auto_crop_flag=not args.no_crop,
        )

    if args.preview:
        preview_gallery(paths)


if __name__ == "__main__":
    main()
