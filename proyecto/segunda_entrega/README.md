# Detector de Piezas Industriales en Tiempo Real

Sistema de visión computacional para detección y localización de piezas
específicas por reconocimiento de features, geometría y estabilidad temporal.

---

## Estructura del proyecto

```
part_detector/
├── config.py           ← Todos los parámetros ajustables
├── part_detector.py    ← Motor de detección (pipeline completo)
├── main.py             ← Entrada CLI (webcam / video / imagen)
├── gallery_builder.py  ← Generador de galería de plantillas rotadas
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
pip install -r requirements.txt
```

> **Nota:** Se requiere `opencv-contrib-python` (no `opencv-python`) para
> acceder a SIFT y AKAZE desde la API de Python.

---

## Uso rápido

### Webcam en tiempo real
```bash
python main.py --ref mi_pieza.png
```

### Desde archivo de video
```bash
python main.py --ref mi_pieza.png --video grabacion.mp4
```

### Imagen estática
```bash
python main.py --ref mi_pieza.png --image foto.jpg
```

### Galería de múltiples referencias (recomendado)
```bash
# Primero generar las plantillas rotadas:
python gallery_builder.py mi_pieza.png --auto --n-angles 8 --output gallery/ --preview

# Luego usar toda la galería:
python main.py --ref gallery/*.png
```

### Cambiar detector en tiempo real
```bash
python main.py --ref pieza.png --detector SIFT
python main.py --ref pieza.png --detector AKAZE   # default — mejor balance
python main.py --ref pieza.png --detector ORB      # más rápido, menos robusto
```

---

## Controles interactivos (ventana de video)

| Tecla | Acción |
|-------|--------|
| `Q` / `ESC` | Salir |
| `S` | Guardar screenshot |
| `D` | Mostrar/ocultar panel de debug |
| `K` | Mostrar/ocultar keypoints |
| `P` | Pausa / continuar |
| `+` / `-` | Subir/bajar umbral de detección |
| `R` | Mostrar/ocultar ROIs de color |

---

## Pipeline completo

```
Frame BGR
    │
    ▼
[1] Prefiltro por color (HSV)
    └── genera ROI mask → limita búsqueda de keypoints
    │
    ▼
[2] Extracción de features (AKAZE / SIFT / ORB)
    └── solo dentro de la ROI mask
    │
    ▼
[3] knnMatch (k=2) + ratio test de Lowe (0.75)
    └── filtra matches ambiguos
    │
    ▼
[4] findHomography con RANSAC
    └── elimina outliers geométricos
    │
    ▼
[5] perspectiveTransform → cuadrilátero + bbox
    └── validación de área y aspecto
    │
    ▼
[6] Score de confianza
    └── α·(inliers/good_matches) + β·(inliers/MAX_REF)
    │
    ▼
[7] Verificación por contorno (matchShapes Hu-moments)
    └── penaliza si contorno no coincide
    │
    ▼
[8] Estabilidad temporal (ventana deslizante + EWA)
    └── confirma detección solo si es consistente N frames
    │
    ▼
DetectionResult { detected, score, stable, quad, bbox, inliers, … }
```

---

## Parámetros clave en `config.py`

### Elegir detector

| Detector | Velocidad | Robustez | Mejor para |
|----------|-----------|----------|------------|
| `AKAZE`  | ★★★★☆    | ★★★★☆   | Uso general — **recomendado** |
| `SIFT`   | ★★☆☆☆    | ★★★★★   | Máxima robustez, sin límite de tiempo |
| `ORB`    | ★★★★★    | ★★★☆☆   | Tiempo real estricto, objetos con textura |

### Ratio test
```python
LOWE_RATIO = 0.75   # más bajo = más estricto (menos matches, más precisos)
                    # rango típico: 0.65 – 0.80
```

### RANSAC
```python
RANSAC_REPROJ_THRESHOLD = 5.0   # píxeles — bajar si la cámara es estable
MIN_GOOD_MATCHES        = 8     # mínimo de buenos matches antes de RANSAC
MIN_INLIERS             = 6     # mínimo de inliers para aceptar la homografía
```

### Score y umbrales
```python
DETECTION_THRESHOLD      = 0.35  # 0–1; subir para menos falsos positivos
CONTOUR_MATCH_THRESHOLD  = 0.35  # Hu-moments; bajar para más estrictez
```

### Prefiltro de color (desactivado por defecto)
```python
# Activar con rangos HSV de la pieza/entorno:
COLOR_RANGES = [
    ([15, 80, 80], [35, 255, 255], "amarillo"),
    ([100, 60, 60], [130, 255, 255], "azul"),
]
```

### Estabilidad temporal
```python
TEMPORAL_WINDOW         = 5   # frames en la ventana
TEMPORAL_MIN_DETECTIONS = 3   # cuántos con score OK para confirmar
TEMPORAL_SCORE_DECAY    = 0.85 # peso exponencial (recientes pesan más)
```

---

## Uso desde código Python

```python
import cv2
from part_detector import PartDetector

# Inicializar
det = PartDetector()

# Cargar una o más referencias
det.add_template("engranaje_recto",  cv2.imread("eng_recto.png"))
det.add_template("engranaje_helico", cv2.imread("eng_helico.png"))

# Procesar frames
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = det.detect(frame)

    for r in results:
        if r.stable:
            print(f"Pieza estable: {r.template_name}  score={r.score:.2f}  "
                  f"centro={r.center}  inliers={r.inliers}")

    PartDetector.draw(frame, results)
    cv2.imshow("demo", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Recomendaciones de plantilla

1. **Recorta ajustado** — elimina fondo innecesario alrededor de la pieza.
2. **Resolución mínima recomendada** — 200×200 px para que AKAZE genere
   suficientes keypoints.
3. **Fondos lisos** — fondo blanco o negro uniforme mejora la extracción.
4. **Para piezas con textura repetitiva** (engranajes, tornillos) — activa
   `USE_CONTOUR_VALIDATION = True` y ajusta `CONTOUR_MATCH_THRESHOLD`.
5. **Múltiples ángulos** — usa `gallery_builder.py --auto --n-angles 8`
   para cubrir rotaciones en plano.
6. **Iluminación variable** — SIFT es más robusto a cambios de iluminación
   que ORB; AKAZE queda en punto medio.

---

## Solución de problemas

| Síntoma | Causa probable | Solución |
|---------|---------------|----------|
| "keypoints insuficientes" | Plantilla muy pequeña o uniforme | Usar imagen más grande; agregar textura; reducir `AKAZE_THRESHOLD` |
| Muchos falsos positivos | Umbral demasiado bajo | Subir `DETECTION_THRESHOLD`; bajar `LOWE_RATIO` |
| Detección inestable/parpadeante | Ventana temporal corta | Subir `TEMPORAL_WINDOW` y `TEMPORAL_MIN_DETECTIONS` |
| Muy lento | SIFT en imagen grande | Cambiar a AKAZE; activar prefiltro de color |
| Bounding box torcido | Homografía con pocos inliers | Subir `MIN_INLIERS`; mejorar iluminación |
