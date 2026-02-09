import cv2
import numpy as np
import time

def orb_detect_and_box(img_gray, tpl_gray, min_good=12):
    # 1) ORB: detecta puntos y descriptores
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(tpl_gray, None)
    kp2, des2 = orb.detectAndCompute(img_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return {"detected": False, "good": 0, "H": None, "box": None}

    # 2) Matcher para descriptores binarios (Hamming)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # 3) knnMatch + ratio test (filtra matches ambiguos)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    ratio = 0.75
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < min_good:
        return {"detected": False, "good": len(good), "H": None, "box": None}

    # 4) Homografía: mapea el template sobre la escena (si hay geometría consistente)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return {"detected": False, "good": len(good), "H": None, "box": None}

    # 5) Bounding box usando la homografía
    h, w = tpl_gray.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H)  # 4 puntos en la escena

    return {"detected": True, "good": len(good), "H": H, "box": projected}

def main():
    THRESH = 0.80   # 0.7-0.9
    scene_path = "vision\\scene.png"
    template_path = "vision\\query.png"

    img = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
    tpl0 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if img is None or tpl0 is None:
        raise FileNotFoundError("No pude leer scene.jpg o template.jpg. Revisa la ruta en assets/.")


    best = {"score": -1.0, "loc": None, "size": None, "scale": None}

    # Rango de escalas: ajústalo a tu caso
    for s in np.linspace(0.5, 1.6, 28):
        tpl = cv2.resize(tpl0, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        th, tw = tpl.shape

        # Evitar templates demasiado chicos o más grandes que la imagen
        if th < 12 or tw < 12 or th >= img.shape[0] or tw >= img.shape[1]:
            continue

        res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best["score"]:
            best.update({"score": float(max_val), "loc": max_loc, "size": (tw, th), "scale": float(s)})

    if best["loc"] is None:
        print("No encontré un match razonable. Ajusta escalas o revisa imágenes.")
        return
    
    # ---- UMBRAL DE DETECCIÓN ----
    if best["score"] < THRESH:
        print(f"No detectado. Mejor score = {best['score']:.3f} < THRESH = {THRESH}")
        return
    else:
        print(f"OBJETO DETECTADO  score = {best['score']:.3f}")

    x, y = best["loc"]
    tw, th = best["size"]

    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (x, y), (x + tw, y + th), (0, 255, 0), 2)
    cv2.putText(out, f"score={best['score']:.3f} scale={best['scale']:.2f}",
                (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print("Mejor match:")
    print(best)

    cv2.imshow("Multi-Scale Template Matching", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()





#Aquí empieza la alternativa lol
scene_path = "vision\\scene.png"
template_path = "vision\\query.png"

img = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
tpl0 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

threshold = 0.6