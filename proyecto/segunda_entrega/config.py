"""
config.py — Parámetros globales del pipeline de detección.
Modifica este archivo para ajustar el comportamiento sin tocar la lógica.
"""

# ──────────────────────────────────────────────
#  1. DETECTOR DE FEATURES (Canal A)
# ──────────────────────────────────────────────
DETECTOR_TYPE = "ORB"           # "SIFT" | "AKAZE" | "ORB"

# SIFT
SIFT_N_FEATURES      = 0
SIFT_N_OCTAVE_LAYERS = 3
SIFT_CONTRAST_THRESH = 0.03
SIFT_EDGE_THRESH     = 12
SIFT_SIGMA           = 1.6

# AKAZE
AKAZE_DESCRIPTOR_TYPE     = 5
AKAZE_DESCRIPTOR_SIZE     = 0
AKAZE_DESCRIPTOR_CHANNELS = 3
AKAZE_THRESHOLD           = 0.001
AKAZE_N_OCTAVES           = 4
AKAZE_N_OCTAVE_LAYERS     = 4

# ORB — ajustado para piezas metálicas lisas
ORB_N_FEATURES   = 5000
ORB_SCALE_FACTOR = 1.15
ORB_N_LEVELS     = 12

# ──────────────────────────────────────────────
#  2. RATIO TEST (Lowe)
# ──────────────────────────────────────────────
LOWE_RATIO = 0.80               # más permisivo para piezas lisas con pocos matches

# ──────────────────────────────────────────────
#  3. RANSAC / HOMOGRAFÍA
# ──────────────────────────────────────────────
RANSAC_REPROJ_THRESHOLD  = 8.0
RANSAC_MAX_ITERS         = 5000
MIN_GOOD_MATCHES         = 4    # bajo porque pieza lisa genera pocos matches
MIN_INLIERS              = 4

# ──────────────────────────────────────────────
#  4. SCORE DE CONFIANZA (Canal A)
# ──────────────────────────────────────────────
SCORE_ALPHA              = 0.5
SCORE_BETA               = 0.5
MAX_INLIERS_REF          = 120
DETECTION_THRESHOLD      = 0.20  # umbral del score FUSIONADO final

# ──────────────────────────────────────────────
#  5. VALIDACIÓN POR TAMAÑO DEL POLÍGONO
# ──────────────────────────────────────────────
MIN_POLYGON_AREA_RATIO   = 0.002
MAX_POLYGON_AREA_RATIO   = 0.95
MAX_ASPECT_RATIO         = 10.0

# ──────────────────────────────────────────────
#  6. PREFILTRO ROI POR COLOR (HSV)
# ──────────────────────────────────────────────
COLOR_RANGES = [
    # Ajustar con el debug HSV de tu cámara real:
    # ([95, 25, 90], [130, 110, 190], "azul_pieza"),
]
COLOR_ROI_MIN_AREA       = 500

# ──────────────────────────────────────────────
#  7. ESTABILIDAD TEMPORAL
# ──────────────────────────────────────────────
TEMPORAL_WINDOW          = 6
TEMPORAL_MIN_DETECTIONS  = 3
TEMPORAL_SCORE_DECAY     = 0.85

# ──────────────────────────────────────────────
#  8. VERIFICACIÓN POR CONTORNO — CANAL A
#     (validación interna del canal de features)
# ──────────────────────────────────────────────
USE_CONTOUR_VALIDATION   = True
CONTOUR_MATCH_THRESHOLD  = 0.80
CONTOUR_CANNY_LOW        = 50
CONTOUR_CANNY_HIGH       = 150

# ──────────────────────────────────────────────
#  9. CANAL B — DETECTOR DE CONTORNO AUTÓNOMO
# ──────────────────────────────────────────────

# ── Preprocesado ──────────────────────────────
# Si True, los umbrales de Canny se calculan automáticamente
# a partir de la mediana del frame (recomendado para iluminación variable)
CONTOUR_CANNY_ADAPTIVE   = True

# ── Filtro de área de candidatos ──────────────
# El contorno candidato debe ocupar entre MIN y MAX del área total del frame
CONTOUR_MIN_AREA_RATIO   = 0.005   # muy pequeño → ignorar; default 0.5% del frame
CONTOUR_MAX_AREA_RATIO   = 0.90    # demasiado grande → ignorar

# Relación de escala permitida entre el área del candidato y el área de referencia
# (0.05 = puede ser 20x más pequeño; 20.0 = puede ser 20x más grande)
CONTOUR_SCALE_MIN        = 0.05
CONTOUR_SCALE_MAX        = 20.0

# ── matchShapes (Hu moments) ──────────────────
# Umbral de matchShapes por encima del cual el score_shape cae a 0
# Valores típicos: 0.1 = muy estricto | 0.3 = normal | 0.8 = permisivo
CONTOUR_HU_THRESHOLD     = 0.30

# Score mínimo de forma para que un candidato pase a la etapa de agujeros
# (evita correr _score_holes en contornos obviamente equivocados)
CONTOUR_MIN_SHAPE_SCORE  = 0.10

# Umbral de score total del canal B para reportar detección autónoma
CONTOUR_DETECTION_THRESHOLD = 0.25

# ── Pesos del score compuesto del Canal B ─────
#    score_B = W_SHAPE·shape + W_SOLIDITY·solidity + W_HOLES·holes
#    Los tres deben sumar 1.0
CONTOUR_W_SHAPE          = 0.55   # forma exterior (Hu moments) — peso mayor
CONTOUR_W_SOLIDITY       = 0.15   # solidez (área vs convex hull) — diferencia cóncavos
CONTOUR_W_HOLES          = 0.30   # agujeros internos — muy discriminativo para esta pieza

# ──────────────────────────────────────────────
#  10. FUSIÓN DUAL-CANAL (A + B)
# ──────────────────────────────────────────────

# Score mínimo de cada canal para considerarlo "activo" en la fusión
FUSION_ORB_MIN_SCORE     = 0.10   # score_orb >= esto → Canal A contribuye
FUSION_CONTOUR_MIN_SCORE = 0.25   # score_contour >= esto → Canal B contribuye

# Pesos de la fusión cuando ambos canales están activos
# score_final = W_ORB·score_orb + W_CONTOUR·score_contour
# Suma 1.0. Para piezas lisas donde ORB falla, dar más peso al contorno.
FUSION_W_ORB             = 0.35
FUSION_W_CONTOUR         = 0.65

# ──────────────────────────────────────────────
#  11. VISUALIZACIÓN
# ──────────────────────────────────────────────
DRAW_KEYPOINTS           = False
DRAW_MATCHES_WINDOW      = False
DRAW_ROI_COLOR           = False
BOX_COLOR_STABLE         = (0, 255, 80)
BOX_COLOR_CANDIDATE      = (0, 200, 255)
BOX_COLOR_WEAK           = (100, 100, 255)
BOX_THICKNESS            = 2
FONT_SCALE               = 0.55
