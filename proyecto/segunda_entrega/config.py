"""
config.py — Parámetros globales del pipeline de detección.
Modifica este archivo para ajustar el comportamiento sin tocar la lógica.
"""

# ──────────────────────────────────────────────
#  1. DETECTOR DE FEATURES
# ──────────────────────────────────────────────
DETECTOR_TYPE = "ORB"          # "SIFT" | "AKAZE" | "ORB"

# SIFT
SIFT_N_FEATURES      = 0         # 0 = sin límite
SIFT_N_OCTAVE_LAYERS = 5
SIFT_CONTRAST_THRESH = 0.02
SIFT_EDGE_THRESH     = 5
SIFT_SIGMA           = 1.6

# AKAZE
AKAZE_DESCRIPTOR_TYPE    = 5     # cv2.AKAZE_DESCRIPTOR_MLDB
AKAZE_DESCRIPTOR_SIZE    = 0
AKAZE_DESCRIPTOR_CHANNELS= 3
AKAZE_THRESHOLD          = 0.001
AKAZE_N_OCTAVES          = 4
AKAZE_N_OCTAVE_LAYERS    = 4

# ORB
ORB_N_FEATURES = 5000
ORB_SCALE_FACTOR= 1.15
ORB_N_LEVELS    = 12

# ──────────────────────────────────────────────
#  2. RATIO TEST (Lowe)
# ──────────────────────────────────────────────
LOWE_RATIO = 0.8                # Valor conservador; 0.8 es el máximo recomendado

# ──────────────────────────────────────────────
#  3. RANSAC / HOMOGRAFÍA
# ──────────────────────────────────────────────
RANSAC_REPROJ_THRESHOLD = 12.0
RANSAC_MAX_ITERS        = 5000  # ORB tiene más outliers, necesita más iteraciones
MIN_GOOD_MATCHES        = 4
MIN_INLIERS             = 4

# ──────────────────────────────────────────────
#  4. SCORE DE CONFIANZA
# ──────────────────────────────────────────────
# Score = α·(inliers/good_matches) + β·(inliers/MAX_INLIERS_REF)
SCORE_ALPHA              = 0.5
SCORE_BETA               = 0.5
MAX_INLIERS_REF          = 120
DETECTION_THRESHOLD      = 0.20  # un poco más permisivo

# ──────────────────────────────────────────────
#  5. VALIDACIÓN POR TAMAÑO DEL POLÍGONO
# ──────────────────────────────────────────────
MIN_POLYGON_AREA_RATIO   = 0.002 # área mínima del bbox como fracción del frame
MAX_POLYGON_AREA_RATIO   = 0.95  # área máxima (evita homografías degeneradas)
MAX_ASPECT_RATIO         = 5.0  # para descartar detecciones muy elongadas

# ──────────────────────────────────────────────
#  6. PREFILTRO ROI POR COLOR (HSV)
# ──────────────────────────────────────────────
# Lista de rangos de color activos. Cada entrada: (lower_hsv, upper_hsv, nombre)
# Deja la lista vacía [] para desactivar el prefiltro y procesar el frame completo.
COLOR_RANGES = [
]
COLOR_ROI_MIN_AREA       = 100   # píxeles² mínimos de la región coloreada

# ──────────────────────────────────────────────
#  7. ESTABILIDAD TEMPORAL
# ──────────────────────────────────────────────
TEMPORAL_WINDOW          = 5     # frames en la ventana deslizante
TEMPORAL_MIN_DETECTIONS  = 3     # cuántos frames con score OK para confirmar
TEMPORAL_SCORE_DECAY     = 0.85  # factor de decaimiento exponencial del score

# ──────────────────────────────────────────────
#  8. VERIFICACIÓN POR CONTORNO (matchShapes)
# ──────────────────────────────────────────────
USE_CONTOUR_VALIDATION   = True
CONTOUR_MATCH_THRESHOLD  = 0.4
CONTOUR_CANNY_LOW        = 50
CONTOUR_CANNY_HIGH       = 150

# ──────────────────────────────────────────────
#  9. VISUALIZACIÓN
# ──────────────────────────────────────────────
DRAW_KEYPOINTS           = False
DRAW_MATCHES_WINDOW      = False # ventana extra con líneas de match
DRAW_ROI_COLOR           = False # mostrar las ROI candidatas en color
BOX_COLOR_STABLE         = (0, 255, 80)   # verde brillante cuando estable
BOX_COLOR_CANDIDATE      = (0, 200, 255)  # amarillo cuando candidato
BOX_COLOR_WEAK           = (100, 100, 255)# rojo suave cuando score bajo
BOX_THICKNESS            = 2
FONT_SCALE               = 0.55
