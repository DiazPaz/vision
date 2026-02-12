import cv2
import numpy as np
import time 
from collections import deque

def main():
    vid = cv2.VideoCapture(0)
    template_path = r"Screenshot 2026-02-09 210243.png"
    
    # Cargar template original
    tpl0 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    tpl0 = cv2.GaussianBlur(tpl0, (5, 5), 1.4)
    
    # Aplicar Canny al template original
    tpl0_canny = cv2.Canny(tpl0, 50, 150)  # Umbral bajo=50, alto=150
    
    scales = np.linspace(0.2, 1.3, 20)
    scaled_templates_gray = []
    scaled_templates_canny = []
    
    # Preparar templates en multi-escala (gris y Canny)
    for s in scales:
        # Template en escala de grises
        tpl_gray = cv2.resize(tpl0, None, fx=float(s), fy=float(s), interpolation=cv2.INTER_AREA)
        th, tw = tpl_gray.shape
        
        # Template Canny escalado
        tpl_canny = cv2.resize(tpl0_canny, None, fx=float(s), fy=float(s), interpolation=cv2.INTER_AREA)
        
        if th >= 12 and tw >= 12:
            scaled_templates_gray.append((float(s), tpl_gray, tw, th))
            scaled_templates_canny.append((float(s), tpl_canny, tw, th))
    
    thresh = 0.2
    
    # Modo de detección: 'gray', 'canny', o 'hybrid'
    detection_mode = 'hybrid'  # Cambia esto para probar diferentes modos
    
    print(f"Modo de detección: {detection_mode}")
    print("Presiona 'm' para cambiar de modo, 'ESC' o 'q' para salir")
    
    while True:
        ret, frame = vid.read()
        if not ret:
            print("No se pudo leer el frame de video.")
            break
        
        # Preprocesamiento del frame
        vid_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vid_blur = cv2.GaussianBlur(vid_gray, (5, 5), 1.4)
        vid_canny = cv2.Canny(vid_blur, 50, 150)  # Aplicar Canny al frame
        
        best_score = -1.0
        best_loc = None
        best_size = None
        best_scale = None
        
        H, W = vid_blur.shape[:2]
        
        # DETECCIÓN SEGÚN EL MODO
        if detection_mode == 'gray':
            # Modo 1: Solo escala de grises (original)
            for s, tpl, tw, th in scaled_templates_gray:
                if th >= H or tw >= W:
                    continue
                res = cv2.matchTemplate(vid_blur, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                
                if max_val > best_score:
                    best_score = float(max_val)
                    best_loc = max_loc
                    best_size = (tw, th)
                    best_scale = float(s)
        
        elif detection_mode == 'canny':
            # Modo 2: Solo bordes Canny
            for s, tpl, tw, th in scaled_templates_canny:
                if th >= H or tw >= W:
                    continue
                res = cv2.matchTemplate(vid_canny, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                
                if max_val > best_score:
                    best_score = float(max_val)
                    best_loc = max_loc
                    best_size = (tw, th)
                    best_scale = float(s)
        
        elif detection_mode == 'hybrid':
            # Modo 3: Híbrido - combina ambos métodos (más robusto)
            for i, (s, tpl_gray, tw, th) in enumerate(scaled_templates_gray):
                if th >= H or tw >= W:
                    continue
                
                # Matching en escala de grises
                res_gray = cv2.matchTemplate(vid_blur, tpl_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val_gray, _, max_loc_gray = cv2.minMaxLoc(res_gray)
                
                # Matching en bordes Canny
                tpl_canny = scaled_templates_canny[i][1]
                res_canny = cv2.matchTemplate(vid_canny, tpl_canny, cv2.TM_CCOEFF_NORMED)
                _, max_val_canny, _, max_loc_canny = cv2.minMaxLoc(res_canny)
                
                # Promedio ponderado (60% gray, 40% canny)
                combined_score = 0.6 * max_val_gray + 0.4 * max_val_canny
                
                if combined_score > best_score:
                    best_score = float(combined_score)
                    best_loc = max_loc_gray  # Usar ubicación de gray
                    best_size = (tw, th)
                    best_scale = float(s)
        
        # Visualización
        out = cv2.cvtColor(vid_blur, cv2.COLOR_GRAY2BGR)
        
        # Mostrar también los bordes Canny en una esquina
        canny_display = cv2.cvtColor(vid_canny, cv2.COLOR_GRAY2BGR)
        canny_small = cv2.resize(canny_display, (W//4, H//4))
        out[10:10+H//4, 10:10+W//4] = canny_small
        
        if best_loc is not None and best_score >= thresh and best_scale >= 0.350:
            x, y = best_loc
            tw, th = best_size
            cv2.rectangle(out, (x, y), (x + tw, y + th), (0, 255, 0), 2)
            cv2.putText(out, f"score={best_score:.3f} scale={best_scale:.2f}",
                        (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(out, f"no detect (best={best_score:.3f} < thr={thresh:.2f})",
                        (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar modo actual
        cv2.putText(out, f"Modo: {detection_mode.upper()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow("Multi-Scale Template Matching + Canny", out)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        elif key == ord("m"):  # Cambiar modo con 'm'
            modes = ['gray', 'canny', 'hybrid']
            current_idx = modes.index(detection_mode)
            detection_mode = modes[(current_idx + 1) % 3]
            print(f"Cambiado a modo: {detection_mode}")
    
    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()