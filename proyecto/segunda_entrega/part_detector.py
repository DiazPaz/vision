"""
part_detector.py — Motor de detección de piezas industriales.

Pipeline completo:
  ROI por color/tamaño → SIFT/AKAZE → knnMatch → ratio test →
  RANSAC homography → cuadrilátero/bbox → validación por contorno →
  estabilidad temporal
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import config


# ══════════════════════════════════════════════════════════════════════════════
#  Estructuras de datos
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionResult:
    """Resultado de detección para un frame."""
    detected:      bool  = False
    score:         float = 0.0
    stable:        bool  = False
    quad:          Optional[np.ndarray] = None   # (4,1,2) float32 — cuadrilátero
    bbox:          Optional[Tuple]      = None   # (x,y,w,h)
    good_matches:  int   = 0
    inliers:       int   = 0
    template_name: str   = ""
    contour_match: float = 1.0  # 0 = perfecto, >threshold = rechazado

    @property
    def center(self) -> Optional[Tuple[int, int]]:
        if self.quad is None:
            return None
        m = self.quad.reshape(-1, 2).mean(axis=0)
        return int(m[0]), int(m[1])


@dataclass
class TemplateInfo:
    """Una plantilla de referencia con sus descriptores pre-calculados."""
    name:        str
    image:       np.ndarray
    gray:        np.ndarray
    mask:        Optional[np.ndarray]
    keypoints:   list
    descriptors: np.ndarray
    contour:     Optional[np.ndarray] = None
    h:           int = 0
    w:           int = 0


# ══════════════════════════════════════════════════════════════════════════════
#  Detector principal
# ══════════════════════════════════════════════════════════════════════════════

class PartDetector:
    """
    Detector de piezas industriales mediante features + verificación geométrica.

    Uso básico:
        det = PartDetector()
        det.add_template("pieza_a", cv2.imread("ref_pieza_a.png"))
        for frame in camera:
            result = det.detect(frame)
            PartDetector.draw(frame, result)
    """

    # ── inicialización ────────────────────────────────────────────────────────

    def __init__(self):
        self._detector  = self._build_detector()
        self._matcher   = self._build_matcher()
        self.templates: List[TemplateInfo] = []

        # ventana de estabilidad temporal por plantilla
        self._temporal_scores: dict = {}   # name → deque de scores
        self._stable_flags:    dict = {}   # name → bool

    # ── API pública ───────────────────────────────────────────────────────────

    def add_template(
        self,
        name:            str,
        image:           np.ndarray,
        mask:            Optional[np.ndarray] = None,
        auto_mask:       bool = False,
        auto_mask_thresh: int = 240,
    ) -> bool:
        """
        Registra una imagen de referencia.

        Args:
            name:             Identificador de la plantilla.
            image:            BGR o escala de grises.
            mask:             Máscara opcional (blanco = región de interés).
            auto_mask:        Si True, genera máscara automática por umbral de brillo.
            auto_mask_thresh: Umbral para auto_mask (píxeles más claros → fondo).
        Returns:
            True si se encontraron keypoints suficientes.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

        if auto_mask and mask is None:
            _, mask = cv2.threshold(gray, auto_mask_thresh, 255, cv2.THRESH_BINARY_INV)

        kp, des = self._detector.detectAndCompute(gray, mask)
        if des is None or len(kp) < config.MIN_GOOD_MATCHES:
            print(f"[PartDetector] ⚠  Plantilla '{name}': solo {len(kp) if kp else 0} "
                  f"keypoints — insuficientes (mín {config.MIN_GOOD_MATCHES}).")
            return False

        # contorno para verificación por forma
        contour = self._extract_reference_contour(gray, mask)

        h, w = gray.shape[:2]
        tmpl = TemplateInfo(
            name=name, image=image, gray=gray, mask=mask,
            keypoints=kp, descriptors=des,
            contour=contour, h=h, w=w,
        )
        self.templates.append(tmpl)
        self._temporal_scores[name] = deque(maxlen=config.TEMPORAL_WINDOW)
        self._stable_flags[name]    = False
        print(f"[PartDetector] ✓  Plantilla '{name}' registrada — {len(kp)} keypoints.")
        return True

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Procesa un frame BGR y devuelve una lista de DetectionResult
        (uno por plantilla registrada, ordenado por score desc).
        """
        if not self.templates:
            return []

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        # ── Paso 2: prefiltro por ROI de color ────────────────────────────────
        roi_masks = self._compute_color_rois(frame, frame_h, frame_w)

        results = []
        for tmpl in self.templates:
            res = self._detect_template(
                tmpl, frame_gray, roi_masks, frame_h, frame_w, frame_area
            )
            self._update_temporal(tmpl.name, res)
            results.append(res)

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ── Pipeline interno ──────────────────────────────────────────────────────

    def _detect_template(
        self,
        tmpl:       TemplateInfo,
        frame_gray: np.ndarray,
        roi_masks:  List[np.ndarray],
        fh: int, fw: int, frame_area: int,
    ) -> DetectionResult:
        base = DetectionResult(template_name=tmpl.name)

        # ── Paso 2b: extraer keypoints solo en ROIs (o frame completo) ────────
        search_mask = None
        if roi_masks:
            combined = np.zeros((fh, fw), dtype=np.uint8)
            for m in roi_masks:
                combined = cv2.bitwise_or(combined, m)
            # dilatar para no recortar keypoints en el borde
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            search_mask = cv2.dilate(combined, kernel)

        kp_frame, des_frame = self._detector.detectAndCompute(frame_gray, search_mask)
        if des_frame is None or len(kp_frame) < config.MIN_GOOD_MATCHES:
            return base

        # ── Paso 3: knnMatch + ratio test ─────────────────────────────────────
        good = self._ratio_test(tmpl.descriptors, des_frame)
        base.good_matches = len(good)
        if len(good) < config.MIN_GOOD_MATCHES:
            return base

        # ── Paso 4: RANSAC homografía ─────────────────────────────────────────
        H, mask_inliers = self._find_homography(good, tmpl.keypoints, kp_frame)
        if H is None:
            return base

        inliers = int(mask_inliers.sum())
        base.inliers = inliers
        if inliers < config.MIN_INLIERS:
            return base

        # ── Paso 5: proyectar bbox de la plantilla ────────────────────────────
        corners_ref = np.float32([
            [0, 0], [tmpl.w, 0], [tmpl.w, tmpl.h], [0, tmpl.h]
        ]).reshape(-1, 1, 2)
        try:
            quad = cv2.perspectiveTransform(corners_ref, H)
        except cv2.error:
            return base

        # Validar área y aspecto del polígono
        if not self._validate_polygon(quad, frame_area):
            return base

        # bbox axis-aligned (para HUD / downstream)
        x, y, bw, bh = cv2.boundingRect(quad.reshape(-1, 1, 2).astype(np.int32))

        # ── Paso 5b: score de confianza ───────────────────────────────────────
        ratio_inliers = inliers / max(base.good_matches, 1)
        ratio_abs     = min(inliers / config.MAX_INLIERS_REF, 1.0)
        score = (config.SCORE_ALPHA * ratio_inliers +
                 config.SCORE_BETA  * ratio_abs)

        if score < config.DETECTION_THRESHOLD:
            return base

        # ── Paso 6: verificación por contorno ────────────────────────────────
        contour_score = 0.0
        if config.USE_CONTOUR_VALIDATION and tmpl.contour is not None:
            contour_score = self._validate_contour(frame_gray, quad, tmpl.contour)
            if contour_score > config.CONTOUR_MATCH_THRESHOLD:
                # penalizar pero no eliminar; deja que el score temporal decida
                score *= 0.5

        base.detected      = True
        base.score         = float(score)
        base.quad          = quad
        base.bbox          = (x, y, bw, bh)
        base.contour_match = float(contour_score)
        return base

    # ── Ratio test (Lowe) ─────────────────────────────────────────────────────

    def _ratio_test(self, des_ref, des_frame) -> list:
        try:
            raw = self._matcher.knnMatch(des_ref, des_frame, k=2)
        except cv2.error:
            return []
        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < config.LOWE_RATIO * n.distance:
                    good.append(m)
        return good

    # ── RANSAC homografía ─────────────────────────────────────────────────────

    def _find_homography(self, good_matches, kp_ref, kp_frame):
        src = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        try:
            H, mask = cv2.findHomography(
                src, dst,
                cv2.RANSAC,
                config.RANSAC_REPROJ_THRESHOLD,
                maxIters=config.RANSAC_MAX_ITERS,
            )
        except cv2.error:
            return None, None
        if H is None or mask is None:
            return None, None
        return H, mask

    # ── Validación de polígono ────────────────────────────────────────────────

    def _validate_polygon(self, quad: np.ndarray, frame_area: int) -> bool:
        pts = quad.reshape(-1, 2)
        area = abs(cv2.contourArea(pts.astype(np.float32)))
        min_a = config.MIN_POLYGON_AREA_RATIO * frame_area
        max_a = config.MAX_POLYGON_AREA_RATIO * frame_area
        if not (min_a < area < max_a):
            return False
        # verificar convexidad razonable
        hull_area = abs(cv2.contourArea(cv2.convexHull(pts.astype(np.float32))))
        if hull_area < 1:
            return False
        # aspecto: lado más largo / más corto
        sides = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
        if min(sides) < 1:
            return False
        if max(sides) / min(sides) > config.MAX_ASPECT_RATIO:
            return False
        return True

    # ── Verificación por contorno ─────────────────────────────────────────────

    def _validate_contour(
        self,
        frame_gray: np.ndarray,
        quad:       np.ndarray,
        ref_contour: np.ndarray,
    ) -> float:
        """Devuelve score Hu-moments (0 = perfecto). Mayor → peor."""
        try:
            mask_roi = np.zeros(frame_gray.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask_roi, [quad.reshape(-1,1,2).astype(np.int32)], 255)
            roi_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=mask_roi)
            edges = cv2.Canny(roi_gray,
                              config.CONTOUR_CANNY_LOW,
                              config.CONTOUR_CANNY_HIGH)
            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return config.CONTOUR_MATCH_THRESHOLD  # neutro
            largest = max(cnts, key=cv2.contourArea)
            score = cv2.matchShapes(ref_contour, largest, cv2.CONTOURS_MATCH_I2, 0)
            return float(score)
        except Exception:
            return config.CONTOUR_MATCH_THRESHOLD

    # ── Extracción de contorno de referencia ──────────────────────────────────

    @staticmethod
    def _extract_reference_contour(
        gray: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        src = gray if mask is None else cv2.bitwise_and(gray, gray, mask=mask)
        edges = cv2.Canny(src,
                          config.CONTOUR_CANNY_LOW,
                          config.CONTOUR_CANNY_HIGH)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        return max(cnts, key=cv2.contourArea)

    # ── Prefiltro por color ───────────────────────────────────────────────────

    @staticmethod
    def _compute_color_rois(
        frame: np.ndarray,
        fh: int, fw: int,
    ) -> List[np.ndarray]:
        if not config.COLOR_RANGES:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masks = []
        for lower, upper, _name in config.COLOR_RANGES:
            lo = np.array(lower, dtype=np.uint8)
            hi = np.array(upper, dtype=np.uint8)
            m  = cv2.inRange(hsv, lo, hi)
            # eliminar ruido
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            # filtrar por área mínima
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean = np.zeros((fh, fw), dtype=np.uint8)
            for c in cnts:
                if cv2.contourArea(c) >= config.COLOR_ROI_MIN_AREA:
                    cv2.drawContours(clean, [c], -1, 255, cv2.FILLED)
            if clean.any():
                masks.append(clean)
        return masks

    # ── Estabilidad temporal ──────────────────────────────────────────────────

    def _update_temporal(self, name: str, res: DetectionResult):
        scores = self._temporal_scores[name]
        scores.append(res.score if res.detected else 0.0)

        # promedio exponencialmente ponderado (más peso a frames recientes)
        weights = np.array([config.TEMPORAL_SCORE_DECAY ** i
                            for i in range(len(scores)-1, -1, -1)])
        weighted_avg = float(np.dot(list(scores), weights) / weights.sum())

        n_above = sum(1 for s in scores if s >= config.DETECTION_THRESHOLD)
        stable  = (n_above >= config.TEMPORAL_MIN_DETECTIONS and
                   weighted_avg >= config.DETECTION_THRESHOLD)

        self._stable_flags[name] = stable
        res.stable = stable

    # ── Builders ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_detector():
        t = config.DETECTOR_TYPE.upper()
        if t == "SIFT":
            return cv2.SIFT_create(
                nfeatures      = config.SIFT_N_FEATURES,
                nOctaveLayers  = config.SIFT_N_OCTAVE_LAYERS,
                contrastThreshold = config.SIFT_CONTRAST_THRESH,
                edgeThreshold  = config.SIFT_EDGE_THRESH,
                sigma          = config.SIFT_SIGMA,
            )
        elif t == "AKAZE":
            return cv2.AKAZE_create(
                descriptor_type     = config.AKAZE_DESCRIPTOR_TYPE,
                descriptor_size     = config.AKAZE_DESCRIPTOR_SIZE,
                descriptor_channels = config.AKAZE_DESCRIPTOR_CHANNELS,
                threshold           = config.AKAZE_THRESHOLD,
                nOctaves            = config.AKAZE_N_OCTAVES,
                nOctaveLayers       = config.AKAZE_N_OCTAVE_LAYERS,
            )
        elif t == "ORB":
            return cv2.ORB_create(
                nfeatures  = config.ORB_N_FEATURES,
                scaleFactor= config.ORB_SCALE_FACTOR,
                nlevels    = config.ORB_N_LEVELS,
            )
        else:
            raise ValueError(f"DETECTOR_TYPE desconocido: '{t}'. Usa SIFT, AKAZE u ORB.")

    @staticmethod
    def _build_matcher():
        t = config.DETECTOR_TYPE.upper()
        if t == "ORB":
            # ORB → descriptores binarios → BFMatcher Hamming
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            # SIFT/AKAZE → float → FLANN más rápido para muchos keypoints
            index_params = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)

    # ── Visualización ─────────────────────────────────────────────────────────

    @staticmethod
    def draw(frame: np.ndarray, results: List[DetectionResult]) -> np.ndarray:
        """
        Dibuja los resultados sobre el frame (in-place).
        Devuelve el mismo frame para encadenado.
        """
        overlay = frame.copy()
        for res in results:
            if not res.detected:
                continue

            # Color del bbox según estabilidad
            if res.stable and res.score >= config.DETECTION_THRESHOLD * 1.5:
                color = config.BOX_COLOR_STABLE
            elif res.stable:
                color = config.BOX_COLOR_CANDIDATE
            else:
                color = config.BOX_COLOR_WEAK

            # Cuadrilátero proyectado
            if res.quad is not None:
                pts = res.quad.reshape(-1, 1, 2).astype(np.int32)
                cv2.polylines(overlay, [pts], True, color, config.BOX_THICKNESS)

                # Relleno semitransparente
                cv2.fillPoly(overlay, [pts.reshape(-1, 2)], color)

            # Centro
            c = res.center
            if c:
                cv2.circle(overlay, c, 5, color, -1)

            # Etiqueta
            lbl = (f"{res.template_name}  "
                   f"sc={res.score:.2f}  "
                   f"in={res.inliers}  "
                   f"{'✓ESTABLE' if res.stable else 'candidato'}")
            if c:
                xt, yt = c[0] + 8, c[1] - 8
                cv2.putText(frame, lbl, (xt, yt),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            config.FONT_SCALE, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, lbl, (xt, yt),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            config.FONT_SCALE, color, 1, cv2.LINE_AA)

        # Mezcla overlay translúcido
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        return frame

    @staticmethod
    def draw_debug_panel(
        frame: np.ndarray,
        results: List[DetectionResult],
        detector_type: str = config.DETECTOR_TYPE,
    ) -> np.ndarray:
        """Panel lateral derecho con estadísticas por plantilla."""
        panel_w = 280
        h, w    = frame.shape[:2]
        panel   = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)

        cv2.putText(panel, f"Detector: {detector_type}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.line(panel, (4, 30), (panel_w-4, 30), (60, 60, 60), 1)

        y = 50
        for res in results:
            det_txt = "ESTABLE" if res.stable else ("detectado" if res.detected else "---")
            col     = (config.BOX_COLOR_STABLE    if res.stable else
                       config.BOX_COLOR_CANDIDATE if res.detected else
                       (80, 80, 80))
            lines = [
                (f"[{res.template_name}]", (220, 220, 220)),
                (f"  Estado : {det_txt}", col),
                (f"  Score  : {res.score:.3f}", col),
                (f"  Inliers: {res.inliers}  GoodM: {res.good_matches}", (160, 160, 160)),
                (f"  Contour: {res.contour_match:.3f}", (160, 160, 160)),
            ]
            for txt, c in lines:
                if y < h - 12:
                    cv2.putText(panel, txt, (8, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
                    y += 16
            y += 4
            if y < h - 12:
                cv2.line(panel, (4, y), (panel_w-4, y), (40, 40, 40), 1)
                y += 8

        return np.hstack([frame, panel])
