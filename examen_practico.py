import cv2
import numpy as np
import os
import time
from collections import deque

# ================================
# CONFIGURACIÓN GENERAL
# ================================

ALGORITMO = "ORB"          # "ORB" o "SIFT"
CAMARA_INDEX = 0
IMAGEN_PATRON = r"VirtualEnv\Vision computacional\pza_fond_blanco.jpeg"

UMBRAL_MATCHES = 50
UMBRAL_RATIO = 0.70
UMBRAL_KP_ESCENA = 60       # evita mandar SCRAP si no hay pieza visible

# Para evitar pulsos falsos en vivo
FRAMES_ESTABLES = 3
VENTANA_DECISION = 10
FRAMES_SIN_PIEZA_RESET = 5
COOLDOWN_SENAL = 2.0        # tiempo mínimo entre pulsos

VENTANA_DECISION = 35          # cuántas muestras usar para decidir
FRAMES_SIN_PIEZA_RESET = 5     # cuántos frames estables sin pieza para reiniciar

# ================================
# CONFIGURACIÓN GPIO / LEDs EN RASPBERRY
# ================================
# LED_DECISION:
#   HIGH = OK
#   LOW  = SCRAP (o aún sin decisión)
#
# LED_PRESENCIA:
#   HIGH = pieza detectada
#   LOW  = no se detecta pieza
#
# IMPORTANTE:
# GPIO -> resistencia 220/330 ohms -> ánodo LED
# cátodo LED -> GND

ENVIAR_SENAL_UR = True

GPIO_DECISION = 22   # BCM | LED 1
GPIO_PRESENCIA = 27  # BCM | LED 2

# ================================
# IMPORT GPIO
# ================================
GPIO_DISPONIBLE = False
if ENVIAR_SENAL_UR:
    try:
        import RPi.GPIO as GPIO
        GPIO_DISPONIBLE = True
    except ImportError:
        print("[UR] RPi.GPIO no encontrado. Se desactiva salida digital.")
        ENVIAR_SENAL_UR = False


# ================================
# CLASE DE INTERFAZ DIGITAL HACIA RASPBERRY
# ================================

class URDigitalBridge:
    def __init__(self, enabled=True, gpio_decision=17, gpio_presencia=27):
        self.enabled = enabled and GPIO_DISPONIBLE
        self.gpio_decision = gpio_decision
        self.gpio_presencia = gpio_presencia

        if not self.enabled:
            print("[GPIO] Salidas digitales deshabilitadas.")
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.gpio_decision, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.gpio_presencia, GPIO.OUT, initial=GPIO.LOW)

        self.reset()
        print("[GPIO] Salidas digitales inicializadas correctamente.")
        print(f"[GPIO] LED_DECISION -> BCM {self.gpio_decision} | HIGH=OK, LOW=SCRAP")
        print(f"[GPIO] LED_PRESENCIA -> BCM {self.gpio_presencia} | HIGH=PIEZA, LOW=NO PIEZA")

    def reset(self):
        if not self.enabled:
            return
        GPIO.output(self.gpio_decision, GPIO.LOW)
        GPIO.output(self.gpio_presencia, GPIO.LOW)

    def set_presencia(self, hay_pieza: bool):
        if not self.enabled:
            return
        GPIO.output(self.gpio_presencia, GPIO.HIGH if hay_pieza else GPIO.LOW)

    def set_decision(self, resultado: str | None):
        if not self.enabled:
            return

        if resultado == "OK":
            GPIO.output(self.gpio_decision, GPIO.HIGH)
        else:
            # LOW tanto para SCRAP como para "sin decisión"
            GPIO.output(self.gpio_decision, GPIO.LOW)

    def set_estado(self, presencia: bool, decision: str | None = None):
        self.set_presencia(presencia)
        self.set_decision(decision)

    def cleanup(self):
        if not self.enabled:
            return
        self.reset()
        GPIO.cleanup()


# ================================
# DETECTOR Y MATCHER
# ================================

def crear_detector(algoritmo: str):
    if algoritmo == "ORB":
        detector = cv2.ORB_create(nfeatures=2000)
        norma = cv2.NORM_HAMMING
        print("[VISION] Detector: ORB | Norma: HAMMING")
    elif algoritmo == "SIFT":
        detector = cv2.SIFT_create(nfeatures=2000)
        norma = cv2.NORM_L2
        print("[VISION] Detector: SIFT | Norma: L2")
    else:
        raise ValueError(f"Algoritmo no reconocido: {algoritmo}")

    matcher = cv2.BFMatcher(norma, crossCheck=False)
    return detector, matcher


# ================================
# EXTRACCIÓN DE CARACTERÍSTICAS
# ================================

def extraer_caracteristicas(detector, imagen_gris):
    kp, des = detector.detectAndCompute(imagen_gris, None)
    return kp, des


# ================================
# MATCHING
# ================================

def hacer_matching(matcher, des_patron, des_actual):
    if des_patron is None or des_actual is None:
        return []
    if len(des_patron) < 2 or len(des_actual) < 2:
        return []

    try:
        matches_knn = matcher.knnMatch(des_patron, des_actual, k=2)
    except Exception as e:
        print(f"[VISION] Error en knnMatch: {e}")
        return []

    buenos = []
    for par in matches_knn:
        if len(par) == 2:
            m, n = par
            if m.distance < UMBRAL_RATIO * n.distance:
                buenos.append(m)

    return buenos


# ================================
# EVALUACIÓN
# ================================

def evaluar(matches_validos, keypoints_actual):
    total_matches = len(matches_validos)
    kp_count = len(keypoints_actual) if keypoints_actual is not None else 0

    if kp_count < UMBRAL_KP_ESCENA:
        resultado = "SIN_PIEZA"
    else:
        resultado = "OK" if total_matches >= UMBRAL_MATCHES else "SCRAP"

    return {
        "matches": total_matches,
        "resultado": resultado,
        "porcentaje": round(total_matches / max(UMBRAL_MATCHES, 1) * 100, 1),
        "kp_actual": kp_count
    }


# ================================
# VISUALIZACIÓN
# ================================

def dibujar_keypoints(imagen, keypoints):
    return cv2.drawKeypoints(
        imagen, keypoints, None,
        color=(0, 230, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )


def dibujar_matches(img_patron_gris, kp_patron,
                    img_actual_gris, kp_actual,
                    matches, evaluacion):
    resultado = evaluacion["resultado"]

    if resultado == "OK":
        color_txt = (0, 220, 0)
    elif resultado == "SCRAP":
        color_txt = (0, 50, 220)
    else:
        color_txt = (0, 220, 220)

    canvas = cv2.drawMatches(
        img_patron_gris, kp_patron,
        img_actual_gris, kp_actual,
        matches[:60], None,
        matchColor=(255, 200, 0),
        singlePointColor=(80, 80, 80),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    alto, ancho = canvas.shape[:2]
    cv2.rectangle(canvas, (0, alto - 65), (ancho, alto), (25, 25, 25), -1)

    linea1 = f"RESULTADO: {resultado}"
    linea2 = f"Matches validos: {evaluacion['matches']} | Umbral: {UMBRAL_MATCHES} | Ratio: {evaluacion['porcentaje']}%"
    linea3 = f"Keypoints escena: {evaluacion['kp_actual']} | Algoritmo: {ALGORITMO}"

    cv2.putText(canvas, linea1, (15, alto - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_txt, 2)
    cv2.putText(canvas, linea2, (15, alto - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
    cv2.putText(canvas, linea3, (15, alto - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

    return canvas


# ================================
# CAPTURA
# ================================

def capturar_imagen(camara_index=0):
    cap = cv2.VideoCapture(camara_index)
    if not cap.isOpened():
        print(f"[VISION] No se puede abrir la cámara (índice {camara_index})")
        return None

    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[VISION] Error al capturar fotograma.")
        return None

    print(f"[VISION] Imagen capturada: {frame.shape[1]}x{frame.shape[0]} px")
    return frame


# ================================
# CARGA DEL PATRÓN
# ================================

def cargar_patron(ruta):
    print("\n[PATRON] Se capturará una NUEVA imagen patrón al iniciar.")
    input("Coloca la PIEZA PATRÓN frente a la cámara y presiona ENTER...")

    img = capturar_imagen(CAMARA_INDEX)
    if img is None:
        raise FileNotFoundError("No se pudo capturar la imagen patrón.")

    # Guardar la nueva referencia sobrescribiendo la anterior
    carpeta = os.path.dirname(ruta)
    if carpeta and not os.path.exists(carpeta):
        os.makedirs(carpeta)

    cv2.imwrite(ruta, img)
    print(f"[PATRON] Nueva referencia guardada en '{ruta}'")

    return img


# ================================
# MODO EN VIVO
# ================================

def modo_en_vivo(detector, matcher, patron_bgr, ur_bridge):
    patron_gris = cv2.cvtColor(patron_bgr, cv2.COLOR_BGR2GRAY)
    kp_p, des_p = extraer_caracteristicas(detector, patron_gris)
    print(f"[PATRON] Keypoints: {len(kp_p)}")

    cap = cv2.VideoCapture(CAMARA_INDEX)
    if not cap.isOpened():
        print("[VISION] No se puede abrir la cámara.")
        return

    congelado = False
    frame_actual = None
    captura_num = 0

    historial_resultados = deque(maxlen=FRAMES_ESTABLES)

    muestras_pieza = []
    decision_final = None
    decision_tomada = False
    frames_sin_pieza = 0

    print("\n[LIVE] Iniciando modo en vivo...")
    print("  ESPACIO = congelar | S = guardar | Q/ESC = salir\n")

    while True:
        if not congelado:
            ret, frame = cap.read()
            if not ret:
                break
            frame_actual = frame.copy()

        gris_actual = cv2.cvtColor(frame_actual, cv2.COLOR_BGR2GRAY)
        kp_a, des_a = extraer_caracteristicas(detector, gris_actual)
        matches = hacer_matching(matcher, des_p, des_a)
        evaluacion = evaluar(matches, kp_a)

        historial_resultados.append(evaluacion["resultado"])
        resultado_estable = None
        if len(historial_resultados) == FRAMES_ESTABLES and len(set(historial_resultados)) == 1:
            resultado_estable = historial_resultados[0]

        if resultado_estable in ("OK", "SCRAP"):
            frames_sin_pieza = 0
            ur_bridge.set_presencia(True)

            if not decision_tomada:
                muestras_pieza.append(resultado_estable)

                if len(muestras_pieza) >= VENTANA_DECISION:
                    total_ok = muestras_pieza.count("OK")
                    total_scrap = muestras_pieza.count("SCRAP")

                    if total_ok > total_scrap:
                        decision_final = "OK"
                    elif total_scrap > total_ok:
                        decision_final = "SCRAP"
                    else:
                        decision_final = "SCRAP"

                    print(f"[DECISION FINAL] {decision_final} | OK={total_ok} | SCRAP={total_scrap}")
                    ur_bridge.set_decision(decision_final)
                    decision_tomada = True

        elif resultado_estable == "SIN_PIEZA":
            frames_sin_pieza += 1

            if frames_sin_pieza >= FRAMES_SIN_PIEZA_RESET:
                if muestras_pieza or decision_tomada or decision_final is not None:
                    print("[LIVE] Pieza retirada. Sistema listo para una nueva decision final.")

                muestras_pieza.clear()
                decision_final = None
                decision_tomada = False
                historial_resultados.clear()
                frames_sin_pieza = 0

                ur_bridge.reset()

        img_kp = dibujar_keypoints(frame_actual.copy(), kp_a)
        estado = "[ CONGELADO ]" if congelado else "[ EN VIVO ]"
        cv2.putText(img_kp, estado, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        if resultado_estable:
            cv2.putText(img_kp, f"Estable: {resultado_estable}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(img_kp, f"Muestras: {len(muestras_pieza)}/{VENTANA_DECISION}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        if decision_final is not None:
            cv2.putText(img_kp, f"Decision final: {decision_final}", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow("Keypoints - Pieza actual", img_kp)

        canvas = dibujar_matches(patron_gris, kp_p, gris_actual, kp_a, matches, evaluacion)
        cv2.imshow("Matches vs Patron", canvas)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord(' '):
            congelado = not congelado
            print(f"[LIVE] {'CONGELADO' if congelado else 'reanudado'}")

        elif tecla in (ord('s'), ord('S')):
            captura_num += 1
            nombre = f"captura_{captura_num:03d}_{evaluacion['resultado']}.jpg"
            cv2.imwrite(nombre, frame_actual)
            print(f"[LIVE] Guardada -> {nombre}")

        elif tecla in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    ur_bridge.reset()


# ================================
# MODO FOTO A FOTO
# ================================

def modo_foto_a_foto(detector, matcher, patron_bgr, ur_bridge):
    patron_gris = cv2.cvtColor(patron_bgr, cv2.COLOR_BGR2GRAY)
    kp_p, des_p = extraer_caracteristicas(detector, patron_gris)
    print(f"[PATRON] Keypoints: {len(kp_p)}")

    historial = []
    ciclo = 0

    while True:
        ciclo += 1
        print(f"\n{'-' * 50}")
        print(f"  CICLO #{ciclo}")
        print(f"{'-' * 50}")

        input("Coloca la pieza y presiona ENTER para capturar...")
        img_actual = capturar_imagen(CAMARA_INDEX)

        if img_actual is None:
            print("[ERROR] No se pudo capturar. Reintenta.")
            continue

        gris_actual = cv2.cvtColor(img_actual, cv2.COLOR_BGR2GRAY)

        kp_a, des_a = extraer_caracteristicas(detector, gris_actual)
        print(f"[VISION] Keypoints detectados: {len(kp_a)}")

        matches = hacer_matching(matcher, des_p, des_a)
        evaluacion = evaluar(matches, kp_a)

        res = evaluacion["resultado"]
        print(f"\n  RESULTADO : {res}")
        print(f"  Matches   : {evaluacion['matches']}")
        print(f"  Umbral    : {UMBRAL_MATCHES}")
        print(f"  Ratio     : {evaluacion['porcentaje']}%")
        print(f"  KP escena : {evaluacion['kp_actual']}")

        historial.append(evaluacion)

        if res in ("OK", "SCRAP"):
            ur_bridge.set_presencia(True)
            ur_bridge.set_decision(res)
        else:
            ur_bridge.set_presencia(False)
            ur_bridge.set_decision(None)
            print("[GPIO] No se detecta pieza. LED de presencia en LOW.")

        img_kp_patron = dibujar_keypoints(patron_bgr.copy(), kp_p)
        img_kp_actual = dibujar_keypoints(img_actual.copy(), kp_a)
        canvas_matches = dibujar_matches(patron_gris, kp_p, gris_actual, kp_a, matches, evaluacion)

        cv2.imshow("Patron - Keypoints", img_kp_patron)
        cv2.imshow("Pieza Actual - Keypoints", img_kp_actual)
        cv2.imshow("Matches vs Patron", canvas_matches)
        print("\n[Presiona cualquier tecla en la ventana para continuar]")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        resp = input("\n¿Inspeccionar otra pieza? [s/n]: ").strip().lower()
        if resp != "s":
            break

    total_ok = sum(1 for r in historial if r["resultado"] == "OK")
    total_scrap = sum(1 for r in historial if r["resultado"] == "SCRAP")
    total_sin = sum(1 for r in historial if r["resultado"] == "SIN_PIEZA")

    print("\n" + "=" * 50)
    print("  REPORTE FINAL")
    print("=" * 50)
    for i, r in enumerate(historial, 1):
        print(f"  Ciclo {i:02d} - {r['resultado']:9s} | Matches: {r['matches']:3d} | KP: {r['kp_actual']:3d}")
    print(f"\n  TOTAL     : {len(historial)}")
    print(f"  OK        : {total_ok}")
    print(f"  SCRAP     : {total_scrap}")
    print(f"  SIN_PIEZA : {total_sin}")
    print("=" * 50)

    ur_bridge.reset()


# ================================
# MAIN
# ================================

def main():
    print("=" * 60)
    print("  SISTEMA DE VISION + LEDs DE ESTADO EN RASPBERRY")
    print("=" * 60)
    print(f"  Algoritmo        : {ALGORITMO}")
    print(f"  Umbral OK        : >= {UMBRAL_MATCHES} matches")
    print(f"  Ratio            : {UMBRAL_RATIO}")
    print(f"  KP min escena    : {UMBRAL_KP_ESCENA}")
    print(f"  Salida GPIO      : {'ACTIVA' if ENVIAR_SENAL_UR else 'DESACTIVADA'}")
    print("=" * 60)

    detector, matcher = crear_detector(ALGORITMO)
    patron_bgr = cargar_patron(IMAGEN_PATRON)
    ur_bridge = URDigitalBridge(
        enabled=ENVIAR_SENAL_UR,
        gpio_decision=GPIO_DECISION,
        gpio_presencia=GPIO_PRESENCIA
    )

    try:
        print("\nModos disponibles:")
        print("[1] En vivo")
        print("[2] Foto a foto")
        opcion = input("\nSelecciona modo [1/2]: ").strip()

        if opcion == "1":
            modo_en_vivo(detector, matcher, patron_bgr, ur_bridge)
        else:
            modo_foto_a_foto(detector, matcher, patron_bgr, ur_bridge)

    finally:
        ur_bridge.cleanup()


if __name__ == "__main__":
    main()