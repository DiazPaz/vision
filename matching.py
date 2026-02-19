import cv2
import numpy as np
import time
from collections import deque

# CPU usage (opcional). Si no esta instalado, el codigo sigue funcionando.
try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False


# =========================
# ORB: version optimizada (template precomputado)
# =========================
def orb_detect_and_box_precomputed(img_gray, orb, tpl_kp, tpl_des, tpl_shape, min_good=12, ratio=0.75):
    kp2, des2 = orb.detectAndCompute(img_gray, None)

    if tpl_des is None or des2 is None or len(tpl_kp) < 4 or len(kp2) < 4:
        return {"detected": False, "good": 0, "H": None, "box": None}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(tpl_des, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < min_good:
        return {"detected": False, "good": len(good), "H": None, "box": None}

    src_pts = np.float32([tpl_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return {"detected": False, "good": len(good), "H": None, "box": None}

    h, w = tpl_shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H)

    return {"detected": True, "good": len(good), "H": H, "box": projected}


# =========================
# Template Matching multi-escala (templates precomputados)
# =========================
def tm_multiscale_best(img_gray, templates_scaled, thresh=0.80):
    # templates_scaled: lista de (scale, tpl_resized)
    best = {"score": -1.0, "loc": None, "size": None, "scale": None}

    for s, tpl in templates_scaled:
        th, tw = tpl.shape
        # si el template no cabe, saltar
        if th >= img_gray.shape[0] or tw >= img_gray.shape[1]:
            continue

        res = cv2.matchTemplate(img_gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best["score"]:
            best.update({"score": float(max_val), "loc": max_loc, "size": (tw, th), "scale": float(s)})

    detected = (best["loc"] is not None) and (best["score"] >= thresh)
    return detected, best


# =========================
# Main en tiempo real con metricas
# =========================
def main():
    # ------- Config -------
    MODE = "TM"  # "TM" o "ORB"
    CAM_INDEX = 0
    FRAME_W, FRAME_H = 640, 480

    THRESH_TM = 0.80
    SCALES = np.linspace(0.5, 1.6, 28)

    ORB_FEATURES = 2000
    ORB_MIN_GOOD = 12
    ORB_RATIO = 0.75

    template_path = r"vision\query.png"

    # ------- Cargar template -------
    tpl0 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if tpl0 is None:
        raise FileNotFoundError(f"No pude leer template: {template_path}")

    # ------- Precomputo TM (ahorra resize por frame) -------
    templates_scaled = []
    for s in SCALES:
        tpl = cv2.resize(tpl0, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        th, tw = tpl.shape
        if th >= 12 and tw >= 12:
            templates_scaled.append((float(s), tpl))

    # ------- Precomputo ORB (ahorra detectar template por frame) -------
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    tpl_kp, tpl_des = orb.detectAndCompute(tpl0, None)
    tpl_shape = tpl0.shape  # (h, w)

    # ------- Camara -------
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        raise RuntimeError("No pude abrir la camara. Revisa CAM_INDEX, permisos, o si otra app la esta usando.")

    # ------- Metric helpers -------
    frame_times = deque(maxlen=30)  # para FPS promedio movil
    last_time = time.perf_counter()

    # CPU
    cpu_percent = 0.0
    last_cpu_t = time.perf_counter()
    proc = psutil.Process() if PSUTIL_OK else None
    if PSUTIL_OK:
        psutil.cpu_percent(None)
        proc.cpu_percent(None)

    # Confusion matrix (por frame) con ground truth manual
    gt_present = False  # toggle con tecla 'p'
    TP = FP = FN = TN = 0

    print("Controles:")
    print("  q: salir")
    print("  m: cambiar modo (TM <-> ORB)")
    print("  p: toggle ground truth (objeto presente/ausente)")
    print("  s: guardar frame actual")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer frame de camara.")
            break

        # Tiempo para FPS
        now = time.perf_counter()
        dt = now - last_time
        last_time = now
        if dt > 0:
            frame_times.append(dt)
        fps = (len(frame_times) / sum(frame_times)) if len(frame_times) > 3 else 0.0

        # Preproceso
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Medir latencia de deteccion
        t0 = time.perf_counter()

        detected = False
        bbox_pts = None
        score_txt = ""

        if MODE.upper() == "TM":
            detected, best = tm_multiscale_best(gray, templates_scaled, thresh=THRESH_TM)
            if detected:
                x, y = best["loc"]
                tw, th = best["size"]
                bbox_pts = np.array([[x, y], [x + tw, y], [x + tw, y + th], [x, y + th]], dtype=np.int32)
                score_txt = f"TM score={best['score']:.3f} sc={best['scale']:.2f}"
            else:
                if best["loc"] is not None:
                    score_txt = f"TM best={best['score']:.3f} (<{THRESH_TM})"
                else:
                    score_txt = "TM no match"

        else:  # ORB
            out_orb = orb_detect_and_box_precomputed(
                gray, orb, tpl_kp, tpl_des, tpl_shape,
                min_good=ORB_MIN_GOOD, ratio=ORB_RATIO
            )
            detected = out_orb["detected"]
            if detected and out_orb["box"] is not None:
                # box viene como (4,1,2)
                pts = out_orb["box"].reshape(-1, 2).astype(np.int32)
                bbox_pts = pts
                score_txt = f"ORB good={out_orb['good']}"
            else:
                score_txt = f"ORB good={out_orb['good']} (<{ORB_MIN_GOOD})"

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # Actualizar confusion matrix por frame usando gt_present
        if detected and gt_present:
            TP += 1
        elif detected and (not gt_present):
            FP += 1
        elif (not detected) and gt_present:
            FN += 1
        else:
            TN += 1

        # Precision aproximada
        precision = (TP / (TP + FP)) if (TP + FP) > 0 else 0.0

        # CPU (aprox, actualizar cada ~0.5s para estabilidad)
        if PSUTIL_OK and (time.perf_counter() - last_cpu_t) > 0.5:
            cpu_percent = psutil.cpu_percent(None)  # total
            # proc_cpu = proc.cpu_percent(None)     # si prefieres proceso
            last_cpu_t = time.perf_counter()

        # Dibujar bbox
        vis = frame.copy()
        if bbox_pts is not None:
            cv2.polylines(vis, [bbox_pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Overlay de metricas
        gt_txt = "GT: PRESENTE" if gt_present else "GT: AUSENTE"
        mode_txt = f"MODO: {MODE}"
        metrics_1 = f"FPS={fps:.1f} | Lat={latency_ms:.1f} ms | CPU={cpu_percent:.0f}%"
        metrics_2 = f"Prec={precision:.3f} | TP={TP} FP={FP} FN={FN} TN={TN}"
        metrics_3 = f"{gt_txt} | {score_txt}"

        y0 = 25
        cv2.putText(vis, mode_txt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, metrics_1, (10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, metrics_2, (10, y0 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, metrics_3, (10, y0 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Realtime Matching + Metricas", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            MODE = "ORB" if MODE.upper() == "TM" else "TM"
        elif key == ord("p"):
            gt_present = not gt_present
        elif key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"capture_{MODE}_{ts}.png"
            cv2.imwrite(fname, vis)
            print(f"[OK] Guardado: {fname}")

    cap.release()
    cv2.destroyAllWindows()

    # Resumen final
    precision = (TP / (TP + FP)) if (TP + FP) > 0 else 0.0
    recall = (TP / (TP + FN)) if (TP + FN) > 0 else 0.0
    print("\n=== Resumen de metricas (aprox) ===")
    print(f"Modo final: {MODE}")
    print(f"TP={TP} FP={FP} FN={FN} TN={TN}")
    print(f"Precision={precision:.4f}")
    print(f"Recall={recall:.4f}")
    if not PSUTIL_OK:
        print("CPU: psutil no instalado. (sudo pip3 install psutil)")

if __name__ == "__main__":
    main()