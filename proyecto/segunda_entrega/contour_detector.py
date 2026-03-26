"""
contour_detector.py — Canal de detección por contorno/silueta.

Funciona como canal paralelo e independiente al de features (ORB/SIFT/AKAZE).
Es especialmente útil para piezas con:
  - Superficie lisa o metálica (pocos keypoints para ORB)
  - Silueta muy característica (agujeros, concavidades, formas compuestas)
  - Fondos con poca textura

Pipeline interno:
  Preprocesado → Canny → findContours (árbol jerárquico) →
  filtro por área → matchShapes (Hu moments) →
  validación por agujeros internos → score compuesto →
  minAreaRect → quad

Firma de forma (ContourSignature):
  Además del contorno externo, se guarda:
    - Momentos de Hu del contorno externo
    - Número de agujeros internos (holes)
    - Hu moments del agujero más grande
    - Solidez (área / área_convexa) — discrimina formas cóncavas
    - Excentricidad aproximada
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import config


# ══════════════════════════════════════════════════════════════════════════════
#  Firma de forma de referencia
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContourSignature:
    """
    Descripción compacta de la forma de una pieza extraída de su plantilla.
    Todos los valores son invariantes a traslación, rotación y escala.
    """
    name:           str

    # ── Contorno externo ──────────────────────────────────────────────────────
    outer_contour:  np.ndarray       # puntos del contorno externo de referencia
    hu_outer:       np.ndarray       # 7 momentos de Hu del exterior
    area_ref:       float            # área en píxeles de la plantilla (para escalar)
    perimeter_ref:  float
    solidity:       float            # area / hull_area; 1.0 = convexo, <1 = cóncavo
    circularity:    float            # 4π·area / perimeter² ; 1.0 = círculo perfecto

    # ── Agujeros internos ─────────────────────────────────────────────────────
    n_holes:        int              # número de contornos internos (huecos)
    hu_largest_hole: Optional[np.ndarray]  # Hu del hueco más grande (si existe)
    area_largest_hole: float         # área del hueco más grande (relativa a outer)

    # ── Aspecto ───────────────────────────────────────────────────────────────
    aspect_ratio:   float            # ancho / alto del bounding rect rotado


# ══════════════════════════════════════════════════════════════════════════════
#  Resultado del canal de contorno
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContourResult:
    """Resultado de una detección por contorno para un frame."""
    detected:         bool  = False
    score:            float = 0.0       # 0–1, mayor es mejor
    score_shape:      float = 0.0       # sub-score de forma exterior (matchShapes)
    score_holes:      float = 0.0       # sub-score de agujeros internos
    score_solidity:   float = 0.0       # sub-score de solidez
    quad:             Optional[np.ndarray] = None   # (4,1,2) int32
    matched_contour:  Optional[np.ndarray] = None   # contorno ganador en el frame
    area_ratio:       float = 0.0       # área_frame / área_referencia
    n_holes_found:    int   = 0


# ══════════════════════════════════════════════════════════════════════════════
#  Detector de contorno
# ══════════════════════════════════════════════════════════════════════════════

class ContourDetector:
    """
    Detector de piezas basado exclusivamente en la silueta/contorno.

    Uso:
        cd = ContourDetector()
        cd.register("pieza", gray_template, mask)

        # por frame:
        result = cd.detect("pieza", frame_gray, frame_area)
    """

    def __init__(self):
        self.signatures: dict[str, ContourSignature] = {}

    # ── API pública ───────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        gray: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Extrae y guarda la firma de forma de una plantilla.
        Devuelve True si el contorno fue encontrado correctamente.
        """
        sig = self._extract_signature(name, gray, mask)
        if sig is None:
            print(f"[ContourDetector] ⚠  '{name}': no se encontró contorno externo.")
            return False
        self.signatures[name] = sig
        print(
            f"[ContourDetector] ✓  '{name}' registrado — "
            f"solidity={sig.solidity:.2f}  holes={sig.n_holes}  "
            f"circ={sig.circularity:.2f}"
        )
        return True

    def detect(
        self,
        name: str,
        frame_gray: np.ndarray,
        frame_area: int,
        roi_mask: Optional[np.ndarray] = None,
    ) -> ContourResult:
        """
        Busca la pieza en el frame usando solo información de contorno.

        Args:
            name:       Nombre de la firma registrada.
            frame_gray: Frame en escala de grises.
            frame_area: Área total del frame en píxeles².
            roi_mask:   Máscara opcional para limitar la búsqueda.
        Returns:
            ContourResult con score y quad si se detectó algo.
        """
        empty = ContourResult()
        if name not in self.signatures:
            return empty
        sig = self.signatures[name]

        # ── 1. Preprocesado ───────────────────────────────────────────────────
        src = frame_gray
        if roi_mask is not None:
            src = cv2.bitwise_and(frame_gray, frame_gray, mask=roi_mask)

        # Bilateral: reduce ruido de textura sin destruir bordes de la pieza
        blurred = cv2.bilateralFilter(src, d=9, sigmaColor=75, sigmaSpace=75)

        # Canny adaptativo si el parámetro está activado, sino usa valores fijos
        if config.CONTOUR_CANNY_ADAPTIVE:
            edges = self._adaptive_canny(blurred)
        else:
            edges = cv2.Canny(blurred,
                              config.CONTOUR_CANNY_LOW,
                              config.CONTOUR_CANNY_HIGH)

        # Cierra pequeñas brechas en los bordes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)

        # ── 2. findContours con árbol jerárquico ──────────────────────────────
        # RETR_TREE devuelve la jerarquía completa: externos e internos (agujeros)
        all_cnts, hierarchy = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if not all_cnts or hierarchy is None:
            return empty

        hierarchy = hierarchy[0]  # shape (N, 4): next, prev, child, parent

        # ── 3. Filtrar candidatos externos por área ───────────────────────────
        # Solo procesar contornos sin padre (nivel superior = silueta exterior)
        candidates = []
        for i, cnt in enumerate(all_cnts):
            if hierarchy[i][3] != -1:
                continue   # tiene padre → es un agujero, no la silueta exterior
            area = cv2.contourArea(cnt)
            # filtrar por área relativa al frame
            if area < config.CONTOUR_MIN_AREA_RATIO * frame_area:
                continue
            if area > config.CONTOUR_MAX_AREA_RATIO * frame_area:
                continue
            # filtrar por escala relativa a la referencia (permite zoom)
            scale_ratio = area / max(sig.area_ref, 1.0)
            if not (config.CONTOUR_SCALE_MIN < scale_ratio < config.CONTOUR_SCALE_MAX):
                continue
            candidates.append((i, cnt, area))

        if not candidates:
            return empty

        # ── 4. Puntuar cada candidato ─────────────────────────────────────────
        best_score  = -1.0
        best_result = empty

        for idx, cnt, area in candidates:
            # ── 4a. matchShapes (Hu moments) — forma exterior ─────────────────
            raw_shape = cv2.matchShapes(sig.outer_contour, cnt,
                                        cv2.CONTOURS_MATCH_I2, 0)
            # Convertir a score 0–1: 0 en threshold → 0 pts; 0 en raw → 1 pt
            score_shape = max(0.0, 1.0 - raw_shape / config.CONTOUR_HU_THRESHOLD)

            if score_shape < config.CONTOUR_MIN_SHAPE_SCORE:
                continue   # forma muy diferente, no vale la pena seguir

            # ── 4b. Solidez ───────────────────────────────────────────────────
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            if hull_area > 0:
                solidity_found = area / hull_area
            else:
                solidity_found = 0.0
            # penalizar diferencia de solidez
            solidity_diff  = abs(sig.solidity - solidity_found)
            score_solidity = max(0.0, 1.0 - solidity_diff / 0.5)

            # ── 4c. Agujeros internos ─────────────────────────────────────────
            holes = self._find_holes(idx, all_cnts, hierarchy)
            score_holes = self._score_holes(sig, holes)

            # ── 4d. Score compuesto ───────────────────────────────────────────
            score = (
                config.CONTOUR_W_SHAPE    * score_shape    +
                config.CONTOUR_W_SOLIDITY * score_solidity +
                config.CONTOUR_W_HOLES    * score_holes
            )

            if score > best_score:
                best_score = score
                # Convertir contorno a quad (minAreaRect → 4 esquinas)
                quad = self._contour_to_quad(cnt)
                best_result = ContourResult(
                    detected        = score >= config.CONTOUR_DETECTION_THRESHOLD,
                    score           = float(score),
                    score_shape     = float(score_shape),
                    score_holes     = float(score_holes),
                    score_solidity  = float(score_solidity),
                    quad            = quad,
                    matched_contour = cnt,
                    area_ratio      = float(area / max(sig.area_ref, 1.0)),
                    n_holes_found   = len(holes),
                )

        return best_result

    # ── Extracción de firma de referencia ─────────────────────────────────────

    def _extract_signature(
        self,
        name: str,
        gray: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Optional[ContourSignature]:

        src = gray if mask is None else cv2.bitwise_and(gray, gray, mask=mask)
        blurred = cv2.bilateralFilter(src, 9, 75, 75)

        if config.CONTOUR_CANNY_ADAPTIVE:
            edges = self._adaptive_canny(blurred)
        else:
            edges = cv2.Canny(blurred,
                              config.CONTOUR_CANNY_LOW,
                              config.CONTOUR_CANNY_HIGH)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        all_cnts, hierarchy = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if not all_cnts or hierarchy is None:
            return None
        hierarchy = hierarchy[0]

        # Tomar el contorno externo más grande
        outer_candidates = [
            (i, c) for i, c in enumerate(all_cnts)
            if hierarchy[i][3] == -1 and cv2.contourArea(c) > 50
        ]
        if not outer_candidates:
            return None
        outer_idx, outer = max(outer_candidates, key=lambda t: cv2.contourArea(t[1]))

        area      = cv2.contourArea(outer)
        perimeter = cv2.arcLength(outer, True)
        hull_area = cv2.contourArea(cv2.convexHull(outer))
        solidity  = area / hull_area if hull_area > 0 else 0.0
        circ      = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        rect      = cv2.minAreaRect(outer)
        rw, rh    = rect[1]
        aspect    = (rw / rh) if rh > 0 else 1.0

        # Hu moments del contorno externo
        hu_outer = cv2.HuMoments(cv2.moments(outer)).flatten()

        # Agujeros: contornos cuyo padre directo es outer_idx
        holes = self._find_holes(outer_idx, all_cnts, hierarchy)
        n_holes = len(holes)

        # Hu del hueco más grande
        hu_hole = None
        area_hole = 0.0
        if holes:
            largest_hole = max(holes, key=cv2.contourArea)
            area_hole    = cv2.contourArea(largest_hole) / max(area, 1.0)
            hu_hole      = cv2.HuMoments(cv2.moments(largest_hole)).flatten()

        return ContourSignature(
            name             = name,
            outer_contour    = outer,
            hu_outer         = hu_outer,
            area_ref         = area,
            perimeter_ref    = perimeter,
            solidity         = solidity,
            circularity      = circ,
            n_holes          = n_holes,
            hu_largest_hole  = hu_hole,
            area_largest_hole= area_hole,
            aspect_ratio     = aspect,
        )

    # ── Scoring de agujeros ───────────────────────────────────────────────────

    @staticmethod
    def _find_holes(
        parent_idx: int,
        all_cnts: tuple,
        hierarchy: np.ndarray,
    ) -> List[np.ndarray]:
        """Devuelve los contornos que son hijos directos de parent_idx."""
        holes = []
        for i, cnt in enumerate(all_cnts):
            if hierarchy[i][3] == parent_idx and cv2.contourArea(cnt) > 20:
                holes.append(cnt)
        return holes

    def _score_holes(
        self,
        sig: ContourSignature,
        holes_found: List[np.ndarray],
    ) -> float:
        """
        Puntúa la coincidencia de agujeros internos entre referencia y frame.
        Combina: número de agujeros + forma del agujero más grande (si existe).
        """
        n_ref   = sig.n_holes
        n_found = len(holes_found)

        # Score por número de agujeros
        if n_ref == 0 and n_found == 0:
            score_count = 1.0
        elif n_ref == 0 or n_found == 0:
            score_count = 0.0
        else:
            # penalizar diferencia relativa
            diff = abs(n_ref - n_found)
            score_count = max(0.0, 1.0 - diff / max(n_ref, 1))

        # Score por forma del agujero más grande
        score_hole_shape = 0.5   # neutro si no hay referencia
        if sig.hu_largest_hole is not None and holes_found:
            largest = max(holes_found, key=cv2.contourArea)
            raw     = cv2.matchShapes(
                sig.outer_contour,   # reutilizamos el contorno externo como proxy
                largest,
                cv2.CONTOURS_MATCH_I2, 0
            )
            score_hole_shape = max(0.0, 1.0 - raw / (config.CONTOUR_HU_THRESHOLD * 2))

        return 0.6 * score_count + 0.4 * score_hole_shape

    # ── Utilidades ────────────────────────────────────────────────────────────

    @staticmethod
    def _contour_to_quad(cnt: np.ndarray) -> np.ndarray:
        """
        Convierte un contorno en un cuadrilátero de 4 puntos usando minAreaRect.
        Devuelve array (4,1,2) int32, compatible con cv2.polylines.
        """
        rect = cv2.minAreaRect(cnt)
        box  = cv2.boxPoints(rect)          # 4 puntos flotantes
        box  = np.int32(box)
        return box.reshape(-1, 1, 2)

    @staticmethod
    def _adaptive_canny(gray: np.ndarray) -> np.ndarray:
        """
        Canny con umbrales calculados automáticamente a partir de la mediana
        de intensidad del frame (funciona mejor con iluminación variable).
        """
        median = float(np.median(gray))
        sigma  = 0.33
        low    = max(0,   int((1.0 - sigma) * median))
        high   = min(255, int((1.0 + sigma) * median))
        return cv2.Canny(gray, low, high)

    # ── Debug ─────────────────────────────────────────────────────────────────

    @staticmethod
    def draw_contour_result(
        frame: np.ndarray,
        result: ContourResult,
        color: tuple = (0, 255, 180),
    ) -> np.ndarray:
        """
        Dibuja el contorno detectado directamente sobre el frame (in-place).
        Se distingue del bbox de ORB porque usa el contorno exacto, no un quad.
        """
        if not result.detected:
            return frame

        if result.matched_contour is not None:
            cv2.drawContours(frame, [result.matched_contour], -1, color, 2)

        if result.quad is not None:
            # esquinas del minAreaRect
            cv2.polylines(frame, [result.quad], True,
                          (255, 255, 0), 1, cv2.LINE_AA)

        return frame
