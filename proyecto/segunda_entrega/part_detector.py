"""
part_detector.py — Motor de detección de piezas industriales.

Pipeline dual-canal:
  Canal A (features):  ROI por color → ORB/SIFT/AKAZE → knnMatch →
                        ratio test → RANSAC homography → quad → score_orb
  Canal B (contorno):  Preprocesado → Canny → findContours (árbol) →
                        filtro área → matchShapes → holes → score_contour
  Fusión:              score_final = W_orb·score_orb + W_cnt·score_contour
  Estabilidad:         ventana temporal con EWA → flag stable
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import config
from contour_detector import ContourDetector, ContourResult


# ══════════════════════════════════════════════════════════════════════════════
#  Estructuras de datos
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionResult:
    """Resultado de detección fusionado (features + contorno) para un frame."""
    detected:      bool  = False
    score:         float = 0.0    # score final fusionado
    score_orb:     float = 0.0    # contribución del canal de features
    score_contour: float = 0.0    # contribución del canal de contorno
    stable:        bool  = False
    source:        str   = "---"  # "ORB" | "CONTOUR" | "FUSION" | "---"

    quad:          Optional[np.ndarray] = None   # (4,1,2) — cuadrilátero de features
    quad_contour:  Optional[np.ndarray] = None   # (4,1,2) — cuadrilátero de contorno
    exact_contour: Optional[np.ndarray] = None   # contorno exacto encontrado en frame
    bbox:          Optional[Tuple]      = None   # (x,y,w,h) axis-aligned

    good_matches:  int   = 0
    inliers:       int   = 0
    template_name: str   = ""
    hu_shape:      float = 1.0   # raw matchShapes (0=perfecto)
    n_holes_found: int   = 0

    @property
    def center(self) -> Optional[Tuple[int, int]]:
        """Centro del quad principal (features si existe, sino contorno)."""
        q = self.quad if self.quad is not None else self.quad_contour
        if q is None:
            return None
        m = q.reshape(-1, 2).mean(axis=0)
        return int(m[0]), int(m[1])


@dataclass
class TemplateInfo:
    """Una plantilla de referencia con sus descriptores pre-calculados."""
    name:        str
    image:       np.ndarray
    gray:        np.ndarray
    mask:        Optional[np.ndarray]
    keypoints:   list
    descriptors: Optional[np.ndarray]
    contour:     Optional[np.ndarray] = None
    h:           int = 0
    w:           int = 0


# ══════════════════════════════════════════════════════════════════════════════
#  Detector principal
# ══════════════════════════════════════════════════════════════════════════════

class PartDetector:
    """
    Detector de piezas industriales con dos canales paralelos:
      - Canal A: features locales (ORB / SIFT / AKAZE) + RANSAC
      - Canal B: contorno/silueta (matchShapes + agujeros internos)

    El canal B actúa como respaldo cuando A falla (piezas lisas/metálicas).
    Ambos scores se fusionan con pesos configurables.
    """

    def __init__(self):
        self._detector        = self._build_detector()
        self._matcher         = self._build_matcher()
        self._contour_det     = ContourDetector()
        self.templates:         List[TemplateInfo] = []
        self._temporal_scores:  dict = {}
        self._stable_flags:     dict = {}

    # ── API pública ───────────────────────────────────────────────────────────

    def add_template(
        self,
        name:             str,
        image:            np.ndarray,
        mask:             Optional[np.ndarray] = None,
        auto_mask:        bool = False,
        auto_mask_thresh: int  = 240,
    ) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

        if auto_mask and mask is None:
            _, mask = cv2.threshold(gray, auto_mask_thresh, 255, cv2.THRESH_BINARY_INV)

        # ── Canal A: features ─────────────────────────────────────────────────
        kp, des = self._detector.detectAndCompute(gray, mask)
        has_features = des is not None and len(kp) >= config.MIN_GOOD_MATCHES
        if not has_features:
            print(f"[PartDetector] ⚠  '{name}': {len(kp) if kp else 0} keypoints — "
                  f"Canal A débil, Canal B tomará el relevo.")

        # Contorno de referencia para verificación en Canal A
        contour = self._extract_reference_contour(gray, mask)

        h, w = gray.shape[:2]
        tmpl = TemplateInfo(
            name=name, image=image, gray=gray, mask=mask,
            keypoints=kp if has_features else [],
            descriptors=des if has_features else None,
            contour=contour, h=h, w=w,
        )
        self.templates.append(tmpl)

        # ── Canal B: contorno ─────────────────────────────────────────────────
        self._contour_det.register(name, gray, mask)

        self._temporal_scores[name] = deque(maxlen=config.TEMPORAL_WINDOW)
        self._stable_flags[name]    = False

        status = f"{len(kp) if kp else 0} kp" if has_features else "sin features"
        print(f"[PartDetector] ✓  '{name}' registrado ({status}) — Canal B activo.")
        return True

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        if not self.templates:
            return []

        frame_gray               = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w         = frame.shape[:2]
        frame_area               = frame_h * frame_w
        roi_masks                = self._compute_color_rois(frame, frame_h, frame_w)

        results = []
        for tmpl in self.templates:
            res = self._detect_template(
                tmpl, frame_gray, roi_masks, frame_h, frame_w, frame_area
            )
            self._update_temporal(tmpl.name, res)
            results.append(res)

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ── Pipeline dual ─────────────────────────────────────────────────────────

    def _detect_template(
        self,
        tmpl:       TemplateInfo,
        frame_gray: np.ndarray,
        roi_masks:  List[np.ndarray],
        fh: int, fw: int, frame_area: int,
    ) -> DetectionResult:

        base = DetectionResult(template_name=tmpl.name)

        # ══ CANAL A — features ════════════════════════════════════════════════
        score_orb = 0.0
        quad_orb  = None
        inliers   = 0
        good_m    = 0

        if tmpl.descriptors is not None:
            score_orb, quad_orb, inliers, good_m = self._run_feature_channel(
                tmpl, frame_gray, roi_masks, fh, fw, frame_area
            )
        base.score_orb    = score_orb
        base.good_matches = good_m
        base.inliers      = inliers
        base.quad         = quad_orb

        # ══ CANAL B — contorno ════════════════════════════════════════════════
        # Construye máscara de búsqueda para el contorno también
        roi_combined = None
        if roi_masks:
            roi_combined = np.zeros((fh, fw), dtype=np.uint8)
            for m in roi_masks:
                roi_combined = cv2.bitwise_or(roi_combined, m)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            roi_combined = cv2.dilate(roi_combined, k)

        cnt_result: ContourResult = self._contour_det.detect(
            tmpl.name, frame_gray, frame_area, roi_combined
        )
        score_contour          = cnt_result.score if cnt_result.detected else 0.0
        base.score_contour     = score_contour
        base.quad_contour      = cnt_result.quad
        base.exact_contour     = cnt_result.matched_contour
        base.hu_shape          = 1.0 - cnt_result.score_shape   # invertir: 0=bueno
        base.n_holes_found     = cnt_result.n_holes_found

        # ══ FUSIÓN ════════════════════════════════════════════════════════════
        fused, source = self._fuse_scores(score_orb, score_contour, quad_orb, cnt_result)
        base.score    = fused
        base.source   = source

        if fused >= config.DETECTION_THRESHOLD:
            base.detected = True
            # Quad principal: preferir features si es sólido, si no usar contorno
            if quad_orb is not None and score_orb >= config.FUSION_ORB_MIN_SCORE:
                base.quad = quad_orb
            elif cnt_result.quad is not None:
                base.quad = cnt_result.quad

            if base.quad is not None:
                x, y, bw, bh = cv2.boundingRect(
                    base.quad.reshape(-1, 1, 2).astype(np.int32)
                )
                base.bbox = (x, y, bw, bh)

        return base

    # ── Canal A: features (ORB/SIFT/AKAZE) ───────────────────────────────────

    def _run_feature_channel(
        self,
        tmpl:       TemplateInfo,
        frame_gray: np.ndarray,
        roi_masks:  List[np.ndarray],
        fh: int, fw: int, frame_area: int,
    ) -> Tuple[float, Optional[np.ndarray], int, int]:
        """Devuelve (score, quad, inliers, good_matches)."""

        search_mask = None
        if roi_masks:
            combined = np.zeros((fh, fw), dtype=np.uint8)
            for m in roi_masks:
                combined = cv2.bitwise_or(combined, m)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            search_mask = cv2.dilate(combined, k)

        kp_f, des_f = self._detector.detectAndCompute(frame_gray, search_mask)
        if des_f is None or len(kp_f) < config.MIN_GOOD_MATCHES:
            return 0.0, None, 0, 0

        good = self._ratio_test(tmpl.descriptors, des_f)
        good_m = len(good)
        if good_m < config.MIN_GOOD_MATCHES:
            return 0.0, None, 0, good_m

        H, mask_in = self._find_homography(good, tmpl.keypoints, kp_f)
        if H is None:
            return 0.0, None, 0, good_m

        inliers = int(mask_in.sum())
        if inliers < config.MIN_INLIERS:
            return 0.0, None, inliers, good_m

        corners_ref = np.float32([
            [0, 0], [tmpl.w, 0], [tmpl.w, tmpl.h], [0, tmpl.h]
        ]).reshape(-1, 1, 2)
        try:
            quad = cv2.perspectiveTransform(corners_ref, H)
        except cv2.error:
            return 0.0, None, inliers, good_m

        if not self._validate_polygon(quad, frame_area):
            return 0.0, None, inliers, good_m

        r_inliers = inliers / max(good_m, 1)
        r_abs     = min(inliers / config.MAX_INLIERS_REF, 1.0)
        score     = config.SCORE_ALPHA * r_inliers + config.SCORE_BETA * r_abs
        return float(score), quad, inliers, good_m

    # ── Fusión de scores ──────────────────────────────────────────────────────

    @staticmethod
    def _fuse_scores(
        score_orb:     float,
        score_contour: float,
        quad_orb:      Optional[np.ndarray],
        cnt_result:    ContourResult,
    ) -> Tuple[float, str]:
        """
        Estrategia de fusión:
          - FUSION:   ambos canales activos → promedio ponderado
          - ORB:      solo Canal A supera su umbral mínimo
          - CONTOUR:  solo Canal B supera su umbral mínimo
          - ---:      ninguno supera umbral mínimo
        """
        orb_ok     = score_orb     >= config.FUSION_ORB_MIN_SCORE
        contour_ok = score_contour >= config.FUSION_CONTOUR_MIN_SCORE

        if orb_ok and contour_ok:
            fused  = (config.FUSION_W_ORB * score_orb +
                      config.FUSION_W_CONTOUR * score_contour)
            source = "FUSION"
        elif orb_ok:
            # Contorno puede añadir un pequeño boost si detectó algo
            boost  = config.FUSION_W_CONTOUR * score_contour * 0.5
            fused  = config.FUSION_W_ORB * score_orb + boost
            source = "ORB"
        elif contour_ok:
            # Features pueden añadir boost menor
            boost  = config.FUSION_W_ORB * score_orb * 0.3
            fused  = config.FUSION_W_CONTOUR * score_contour + boost
            source = "CONTOUR"
        else:
            fused  = max(score_orb, score_contour) * 0.5
            source = "---"

        return float(fused), source

    # ── Ratio test ────────────────────────────────────────────────────────────

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

    @staticmethod
    def _find_homography(good_matches, kp_ref, kp_frame):
        src = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        try:
            H, mask = cv2.findHomography(
                src, dst, cv2.RANSAC,
                config.RANSAC_REPROJ_THRESHOLD,
                maxIters=config.RANSAC_MAX_ITERS,
            )
        except cv2.error:
            return None, None
        if H is None or mask is None:
            return None, None
        return H, mask

    # ── Validación de polígono ────────────────────────────────────────────────

    @staticmethod
    def _validate_polygon(quad: np.ndarray, frame_area: int) -> bool:
        pts      = quad.reshape(-1, 2)
        area     = abs(cv2.contourArea(pts.astype(np.float32)))
        min_a    = config.MIN_POLYGON_AREA_RATIO * frame_area
        max_a    = config.MAX_POLYGON_AREA_RATIO * frame_area
        if not (min_a < area < max_a):
            return False
        hull_a = abs(cv2.contourArea(cv2.convexHull(pts.astype(np.float32))))
        if hull_a < 1:
            return False
        sides = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
        if min(sides) < 1:
            return False
        if max(sides) / min(sides) > config.MAX_ASPECT_RATIO:
            return False
        return True

    # ── Extracción de contorno de referencia (para Canal A) ───────────────────

    @staticmethod
    def _extract_reference_contour(
        gray: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        src   = gray if mask is None else cv2.bitwise_and(gray, gray, mask=mask)
        edges = cv2.Canny(src, config.CONTOUR_CANNY_LOW, config.CONTOUR_CANNY_HIGH)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        return max(cnts, key=cv2.contourArea)

    # ── Prefiltro por color ───────────────────────────────────────────────────

    @staticmethod
    def _compute_color_rois(frame, fh, fw) -> List[np.ndarray]:
        if not config.COLOR_RANGES:
            return []
        hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        masks  = []
        for lower, upper, _name in config.COLOR_RANGES:
            m = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean   = np.zeros((fh, fw), dtype=np.uint8)
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
        w       = np.array([config.TEMPORAL_SCORE_DECAY**i
                            for i in range(len(scores)-1, -1, -1)])
        w_avg   = float(np.dot(list(scores), w) / w.sum())
        n_above = sum(1 for s in scores if s >= config.DETECTION_THRESHOLD)
        stable  = (n_above >= config.TEMPORAL_MIN_DETECTIONS and
                   w_avg   >= config.DETECTION_THRESHOLD)
        self._stable_flags[name] = stable
        res.stable = stable

    # ── Builders ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_detector():
        t = config.DETECTOR_TYPE.upper()
        if t == "SIFT":
            return cv2.SIFT_create(
                nfeatures=config.SIFT_N_FEATURES,
                nOctaveLayers=config.SIFT_N_OCTAVE_LAYERS,
                contrastThreshold=config.SIFT_CONTRAST_THRESH,
                edgeThreshold=config.SIFT_EDGE_THRESH,
                sigma=config.SIFT_SIGMA,
            )
        elif t == "AKAZE":
            return cv2.AKAZE_create(
                descriptor_type=config.AKAZE_DESCRIPTOR_TYPE,
                descriptor_size=config.AKAZE_DESCRIPTOR_SIZE,
                descriptor_channels=config.AKAZE_DESCRIPTOR_CHANNELS,
                threshold=config.AKAZE_THRESHOLD,
                nOctaves=config.AKAZE_N_OCTAVES,
                nOctaveLayers=config.AKAZE_N_OCTAVE_LAYERS,
            )
        elif t == "ORB":
            return cv2.ORB_create(
                nfeatures=config.ORB_N_FEATURES,
                scaleFactor=config.ORB_SCALE_FACTOR,
                nlevels=config.ORB_N_LEVELS,
            )
        else:
            raise ValueError(f"DETECTOR_TYPE desconocido: '{t}'.")

    @staticmethod
    def _build_matcher():
        t = config.DETECTOR_TYPE.upper()
        if t == "ORB":
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            idx_p  = dict(algorithm=1, trees=5)
            srch_p = dict(checks=50)
            return cv2.FlannBasedMatcher(idx_p, srch_p)

    # ── Visualización ─────────────────────────────────────────────────────────

    @staticmethod
    def draw(frame: np.ndarray, results: List[DetectionResult]) -> np.ndarray:
        overlay = frame.copy()
        for res in results:
            if not res.detected:
                continue

            if res.stable and res.score >= config.DETECTION_THRESHOLD * 1.5:
                color = config.BOX_COLOR_STABLE
            elif res.stable:
                color = config.BOX_COLOR_CANDIDATE
            else:
                color = config.BOX_COLOR_WEAK

            # ── Contorno exacto (Canal B) — dibujado primero, más interno ────
            if res.exact_contour is not None:
                cnt_color = tuple(min(255, c + 60) for c in color)
                cv2.drawContours(overlay, [res.exact_contour], -1, cnt_color, 2)

            # ── Quad de features (Canal A) ────────────────────────────────────
            if res.quad is not None:
                pts = res.quad.reshape(-1, 1, 2).astype(np.int32)
                cv2.polylines(overlay, [pts], True, color, config.BOX_THICKNESS)
                cv2.fillPoly(overlay, [pts.reshape(-1, 2)], color)

            # ── Quad de contorno (Canal B, si difiere del A) ──────────────────
            if res.quad_contour is not None and res.quad is None:
                pts = res.quad_contour.reshape(-1, 1, 2).astype(np.int32)
                cv2.polylines(overlay, [pts], True, color, config.BOX_THICKNESS)
                cv2.fillPoly(overlay, [pts.reshape(-1, 2)], color)

            # ── Centro + etiqueta ─────────────────────────────────────────────
            c = res.center
            if c:
                cv2.circle(overlay, c, 5, color, -1)
                lbl = (f"{res.template_name}  "
                       f"sc={res.score:.2f}  [{res.source}]  "
                       f"{'✓ESTABLE' if res.stable else 'candidato'}")
                xt, yt = c[0] + 8, c[1] - 8
                cv2.putText(frame, lbl, (xt, yt),
                            cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, lbl, (xt, yt),
                            cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                            color, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        return frame

    @staticmethod
    def draw_debug_panel(
        frame: np.ndarray,
        results: List[DetectionResult],
        detector_type: str = config.DETECTOR_TYPE,
    ) -> np.ndarray:
        panel_w = 310
        h, w    = frame.shape[:2]
        panel   = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (18, 18, 18)

        cv2.putText(panel, f"Detector: {detector_type} + CONTOUR",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.line(panel, (4, 30), (panel_w-4, 30), (55, 55, 55), 1)

        SOURCE_COLORS = {
            "FUSION":   (0, 255, 180),
            "CONTOUR":  (255, 200, 0),
            "ORB":      (100, 200, 255),
            "---":      (70, 70, 70),
        }

        y = 50
        for res in results:
            det_txt = "ESTABLE" if res.stable else ("detectado" if res.detected else "---")
            s_col   = SOURCE_COLORS.get(res.source, (120, 120, 120))
            col     = (config.BOX_COLOR_STABLE    if res.stable else
                       config.BOX_COLOR_CANDIDATE if res.detected else (70, 70, 70))
            lines = [
                (f"[{res.template_name}]",                        (220, 220, 220)),
                (f"  Estado  : {det_txt}",                        col),
                (f"  Fuente  : {res.source}",                     s_col),
                (f"  Score   : {res.score:.3f}",                  col),
                (f"  ORB     : {res.score_orb:.3f}  "
                 f"in={res.inliers} gm={res.good_matches}",       (140, 180, 255)),
                (f"  Contorno: {res.score_contour:.3f}  "
                 f"holes={res.n_holes_found}",                     (255, 200, 80)),
                (f"  Hu shape: {res.hu_shape:.3f}",               (160, 160, 160)),
            ]
            for txt, c in lines:
                if y < h - 12:
                    cv2.putText(panel, txt, (8, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, c, 1, cv2.LINE_AA)
                    y += 15
            y += 4
            if y < h - 12:
                cv2.line(panel, (4, y), (panel_w-4, y), (38, 38, 38), 1)
                y += 8

        return np.hstack([frame, panel])
