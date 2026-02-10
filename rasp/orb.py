import cv2
import numpy as np
import time

# Configuración
WIN_NAME = "Camera Matching"
CAM_INDEX = 0
MIN_MATCH = 20  # Mínimo de matches para considerar detección válida
MAX_FEATURES = 2000
RATIO_THRESHOLD = 0.75  # Más restrictivo para mejores matches
RANSAC_THRESHOLD = 5.0
MAX_MATCHES_DRAW = 75

def compute_template_features(detector, img_bgr):
    """Extrae características de la imagen template."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(gray, None)
    return gray, kp, des

def create_detector():
    """Crea detector ORB optimizado."""
    return cv2.ORB_create(
        nfeatures=MAX_FEATURES,
        scaleFactor=1.2,
        nlevels=10,
        edgeThreshold=15,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_FAST_SCORE,
        patchSize=31,
        fastThreshold=30
    )

def create_matcher():
    """Crea matcher FLANN optimizado para ORB."""
    index_params = dict(
        algorithm=6,  # FLANN_INDEX_LSH
        table_number=12,  # Aumentado para mejor precisión
        key_size=20,      # Aumentado para mejor precisión
        multi_probe_level=2
    )
    search_params = dict(checks=200)  # Aumentado para mejor calidad
    return cv2.FlannBasedMatcher(index_params, search_params)

def apply_ratio_test(knn_matches, ratio=0.75):
    """Aplica Lowe's ratio test de forma eficiente."""
    good_matches = []
    for match in knn_matches:
        if len(match) == 2:
            m, n = match
            if m.distance < ratio * n.distance:
                good_matches.append(m)
    return good_matches

def find_object_homography(kp1, kp2, matches, min_matches=MIN_MATCH):
    """Encuentra homografía y calcula inliers."""
    if len(matches) < min_matches:
        return None, None, 0
    
    # Extraer puntos
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Calcular homografía con RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
    
    inliers = int(mask.sum()) if mask is not None else 0
    
    return H, mask, inliers

def get_transformed_corners(H, width, height):
    """Calcula las esquinas transformadas del template."""
    if H is None:
        return None
    
    # Esquinas del template
    corners = np.float32([
        [[0, 0]], 
        [[width - 1, 0]], 
        [[width - 1, height - 1]], 
        [[0, height - 1]]
    ])
    
    # Transformar esquinas
    try:
        transformed = cv2.perspectiveTransform(corners, H)
        return transformed
    except cv2.error:
        return None

def draw_bounding_box_on_combined(result_img, transformed_corners, offset_x):
    """
    Dibuja bounding box en la imagen combinada (drawMatches result).
    
    Args:
        result_img: Imagen resultado de cv2.drawMatches
        transformed_corners: Esquinas transformadas del objeto
        offset_x: Offset horizontal (ancho de la imagen template)
    """
    if transformed_corners is None:
        return False
    
    # Ajustar coordenadas para la imagen combinada
    # drawMatches coloca la imagen template a la izquierda y el frame a la derecha
    adjusted_corners = transformed_corners.copy()
    adjusted_corners[:, :, 0] += offset_x  # Desplazar X por el ancho del template
    
    # Verificar que las coordenadas sean válidas
    if np.any(np.isnan(adjusted_corners)) or np.any(np.isinf(adjusted_corners)):
        return False
    
    # Dibujar polígono
    cv2.polylines(result_img, [np.int32(adjusted_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
    
    # Opcional: dibujar puntos en las esquinas para debug
    for corner in adjusted_corners:
        pt = tuple(corner[0].astype(int))
        cv2.circle(result_img, pt, 5, (0, 0, 255), -1)
    
    return True

def setup_camera(cam_index=CAM_INDEX):
    """Configura la cámara con parámetros optimizados."""
    cap = cv2.VideoCapture(cam_index)
    
    # Configuración de resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Optimizaciones adicionales
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latencia
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    return cap

def main():
    # Cargar imagen template
    img1 = cv2.imread(r"vision\rasp\rubik.png")
    if img1 is None:
        print("Error: No se pudo leer vision\\rasp\\rubik.png")
        return

    # Crear detector y matcher
    detector = create_detector()
    matcher = create_matcher()

    # Extraer características del template
    gray1, kp1, desc1 = compute_template_features(detector, img1)
    if desc1 is None or len(kp1) < MIN_MATCH:
        print("Error: Muy pocos keypoints en el template")
        return

    h1, w1 = gray1.shape[:2]
    print(f"Template cargado: {len(kp1)} keypoints detectados")

    # Configurar cámara
    cap = setup_camera()
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        return

    # Variables para FPS
    last_t = time.perf_counter()
    fps = 0.0

    print("Sistema iniciado. Presiona 'q' o ESC para salir, ESPACIO para nueva ROI")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error al leer frame")
                break

            # Calcular FPS
            t0 = time.perf_counter()
            dt = t0 - last_t
            last_t = t0
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 and dt > 0 else (1.0 / dt if dt > 0 else 0)

            # Convertir a escala de grises
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar características
            kp2, desc2 = detector.detectAndCompute(gray2, None)

            # Verificar si hay descriptores válidos
            if desc2 is None or len(kp2) < MIN_MATCH:
                # Mostrar frame sin matches
                status = f"Insuficientes features | FPS: {fps:.1f}"
                cv2.putText(frame, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(WIN_NAME, frame)
            else:
                # Hacer matching
                knn_matches = matcher.knnMatch(desc1, desc2, k=2)
                
                # Aplicar ratio test
                good_matches = apply_ratio_test(knn_matches, RATIO_THRESHOLD)

                # Encontrar homografía
                H, mask, inliers = find_object_homography(kp1, kp2, good_matches)

                # Calcular esquinas transformadas
                transformed_corners = get_transformed_corners(H, w1, h1)
                
                # Preparar mask para drawMatches
                if mask is not None:
                    matches_mask = mask.ravel().astype(int).tolist()
                else:
                    matches_mask = [1] * len(good_matches)
                
                # Limitar número de matches a dibujar
                num_draw = min(MAX_MATCHES_DRAW, len(good_matches))
                
                # Dibujar matches (crea imagen combinada: template | frame)
                result = cv2.drawMatches(
                    img1, kp1,
                    frame, kp2,  # Sin copiar el frame
                    good_matches[:num_draw],
                    None,
                    matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                    matchesMask=matches_mask[:num_draw],
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                
                # AQUÍ está la solución: Dibujar el bounding box DESPUÉS de drawMatches
                # y ajustar las coordenadas por el ancho del template
                bbox_drawn = False
                if transformed_corners is not None and inliers >= MIN_MATCH:
                    bbox_drawn = draw_bounding_box_on_combined(result, transformed_corners, w1)
                
                # Agregar información
                status = f"Matches: {len(good_matches)} | Inliers: {inliers} | BBox: {'OK' if bbox_drawn else 'NO'} | FPS: {fps:.1f}"
                color = (0, 255, 0) if bbox_drawn else (0, 255, 255)
                cv2.putText(result, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow(WIN_NAME, result)

            # Manejo de teclas
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):  # ESC o 'q'
                break
            elif key == ord(" "):  # ESPACIO - capturar nueva ROI
                print("Selecciona una región de interés...")
                roi = cv2.selectROI(WIN_NAME, frame, fromCenter=False, showCrosshair=True)
                x, y, w, h = roi
                
                if w > 0 and h > 0:
                    # Extraer ROI
                    img1 = frame[y:y+h, x:x+w].copy()
                    
                    # Extraer nuevas características
                    gray1, kp1, desc1 = compute_template_features(detector, img1)
                    
                    if desc1 is None or len(kp1) < MIN_MATCH:
                        print(f"Error: ROI con muy pocos keypoints ({len(kp1) if kp1 else 0})")
                        print("Intenta seleccionar una región con más detalles")
                    else:
                        h1, w1 = gray1.shape[:2]
                        print(f"✓ Nueva referencia: {len(kp1)} keypoints | Tamaño: {w1}x{h1}")
                else:
                    print("Selección cancelada")

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario")
    finally:
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        print("Recursos liberados")

if __name__ == "__main__":
    main()