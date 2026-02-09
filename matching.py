import cv2
import numpy as np

def main():
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