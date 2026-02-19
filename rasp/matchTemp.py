import cv2
import numpy as np
import time
from collections import deque

# ---------------- GPIO (Raspberry Pi) ----------------
# If you are NOT on a Raspberry Pi, the code will still run without GPIO.
try:
    from gpiozero import LED
    GPIO_OK = True
except Exception:
    GPIO_OK = False

# ---------------- CPU (optional) ----------------
# If psutil is not installed, code still runs without CPU metric.
try:
    import psutil
    PSUTIL_OK = True
except Exception:
    PSUTIL_OK = False

LED_GPIO_PIN = 17  # change if you wired the LED to another GPIO
LED_ON_TIME_S = 0.15  # how long to keep LED on when detection happens

def main():
    # LED setup
    led = None
    led_off_deadline = 0.0
    last_trigger_time = 0.0
    trigger_cooldown_s = 0.25  # avoid re-triggering every single frame

    if GPIO_OK:
        led = LED(LED_GPIO_PIN)
        led.off()
        print(f"GPIO OK. LED on GPIO{LED_GPIO_PIN}")
    else:
        print("GPIO not available (running without LED control). Install gpiozero or run on Raspberry Pi.")

    # ---------------- Metrics state ----------------
    # FPS: moving average over last N frames
    frame_dt = deque(maxlen=30)
    last_frame_t = time.perf_counter()

    # Precision & False Positives require ground truth (manual toggle)
    gt_present = False  # press 'p' to toggle "object is present"
    TP = FP = FN = TN = 0

    # CPU metric
    cpu_percent = 0.0
    last_cpu_t = time.perf_counter()
    if PSUTIL_OK:
        psutil.cpu_percent(None)  # warm up

    # Latency metric
    latency_ms = 0.0

    print("Controls: 'm' change mode | 'p' toggle GT present/absent | 'q' or ESC quit")

    vid = cv2.VideoCapture(0)
    template_path = r"WIN_20260212_15_18_24_Pro.jpg"

    # Load original template
    tpl0 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if tpl0 is None:
        print(f"Template not found: {template_path}")
        return

    tpl0 = cv2.GaussianBlur(tpl0, (5, 5), 1.4)

    # Canny on template
    tpl0_canny = cv2.Canny(tpl0, 50, 150)

    scales = np.linspace(0.2, 1.3, 20)
    scaled_templates_gray = []
    scaled_templates_canny = []

    # Prepare multi-scale templates (gray + canny)
    for s in scales:
        tpl_gray = cv2.resize(tpl0, None, fx=float(s), fy=float(s), interpolation=cv2.INTER_AREA)
        th, tw = tpl_gray.shape

        tpl_canny = cv2.resize(tpl0_canny, None, fx=float(s), fy=float(s), interpolation=cv2.INTER_AREA)

        if th >= 12 and tw >= 12:
            scaled_templates_gray.append((float(s), tpl_gray, tw, th))
            scaled_templates_canny.append((float(s), tpl_canny, tw, th))

    thresh = 0.75
    detection_mode = "hybrid"

    print(f"Detection mode: {detection_mode}")
    print("Press 'm' to change mode, 'p' to toggle GT, 'ESC' or 'q' to quit")

    while True:
        ret, frame = vid.read()
        if not ret:
            print("Could not read video frame.")
            break

        # ---------------- FPS ----------------
        now_t = time.perf_counter()
        dt = now_t - last_frame_t
        last_frame_t = now_t
        if dt > 0:
            frame_dt.append(dt)
        fps = (len(frame_dt) / sum(frame_dt)) if len(frame_dt) > 3 else 0.0

        now = time.time()

        # Frame preprocessing
        vid_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vid_blur = cv2.GaussianBlur(vid_gray, (5, 5), 1.4)
        vid_canny = cv2.Canny(vid_blur, 50, 150)

        best_score = -1.0
        best_loc = None
        best_size = None
        best_scale = None

        H, W = vid_blur.shape[:2]

        # ---------------- Latency (only detection block) ----------------
        t0 = time.perf_counter()

        if detection_mode == "gray":
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

        elif detection_mode == "canny":
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

        elif detection_mode == "hybrid":
            for i, (s, tpl_gray, tw, th) in enumerate(scaled_templates_gray):
                if th >= H or tw >= W:
                    continue

                res_gray = cv2.matchTemplate(vid_blur, tpl_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val_gray, _, max_loc_gray = cv2.minMaxLoc(res_gray)

                tpl_canny = scaled_templates_canny[i][1]
                res_canny = cv2.matchTemplate(vid_canny, tpl_canny, cv2.TM_CCOEFF_NORMED)
                _, max_val_canny, _, _ = cv2.minMaxLoc(res_canny)

                combined_score = 0.6 * max_val_gray + 0.4 * max_val_canny

                if combined_score > best_score:
                    best_score = float(combined_score)
                    best_loc = max_loc_gray
                    best_size = (tw, th)
                    best_scale = float(s)

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # Visualization
        out = cv2.cvtColor(vid_blur, cv2.COLOR_GRAY2BGR)

        canny_display = cv2.cvtColor(vid_canny, cv2.COLOR_GRAY2BGR)
        canny_small = cv2.resize(canny_display, (W // 4, H // 4))
        out[10:10 + H // 4, 10:10 + W // 4] = canny_small

        detected = (best_loc is not None and best_score >= thresh and (best_scale is not None and best_scale >= 0.1))

        # ---------------- Confusion matrix for Precision / False positives (manual GT) ----------------
        if detected and gt_present:
            TP += 1
        elif detected and not gt_present:
            FP += 1
        elif (not detected) and gt_present:
            FN += 1
        else:
            TN += 1

        precision = (TP / (TP + FP)) if (TP + FP) > 0 else 0.0

        # ---------------- CPU (approx, update every 0.5s) ----------------
        if PSUTIL_OK and (time.perf_counter() - last_cpu_t) > 0.5:
            cpu_percent = psutil.cpu_percent(None)
            last_cpu_t = time.perf_counter()

        if detected:
            x, y = best_loc
            tw, th = best_size
            cv2.rectangle(out, (x, y), (x + tw, y + th), (0, 255, 0), 2)
            cv2.putText(out, f"score={best_score:.3f} scale={best_scale:.2f}",
                        (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # ---------- LED trigger when bounding box is detected ----------
            if led is not None and (now - last_trigger_time) >= trigger_cooldown_s:
                led.on()
                led_off_deadline = now + LED_ON_TIME_S
                last_trigger_time = now
        else:
            cv2.putText(out, f"no detect (best={best_score:.3f} < thr={thresh:.2f})",
                        (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Turn off LED after a short pulse
        if led is not None and led_off_deadline > 0 and now >= led_off_deadline:
            led.off()
            led_off_deadline = 0.0

        # ---------------- On-screen metrics overlay ----------------
        gt_txt = "GT:PRESENT" if gt_present else "GT:ABSENT"
        line1 = f"Mode: {detection_mode.upper()} | FPS: {fps:.1f} | Lat: {latency_ms:.1f} ms"
        line2 = f"Prec: {precision:.3f} | FP: {FP} | TP:{TP} FN:{FN} TN:{TN}"
        if PSUTIL_OK:
            line3 = f"CPU: {cpu_percent:.0f}% | {gt_txt} | thr:{thresh:.2f}"
        else:
            line3 = f"CPU: N/A (pip3 install psutil) | {gt_txt} | thr:{thresh:.2f}"

        cv2.putText(out, line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.putText(out, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.putText(out, line3, (10, 71), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        cv2.imshow("Multi-Scale Template Matching + Canny", out)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        elif key == ord("m"):
            modes = ["gray", "canny", "hybrid"]
            current_idx = modes.index(detection_mode)
            detection_mode = modes[(current_idx + 1) % 3]
            print(f"Switched to mode: {detection_mode}")
        elif key == ord("p"):
            gt_present = not gt_present
            print(f"GT toggled -> {'PRESENT' if gt_present else 'ABSENT'}")

    # Cleanup
    vid.release()
    cv2.destroyAllWindows()
    if led is not None:
        led.off()

    # Final summary (console)
    precision = (TP / (TP + FP)) if (TP + FP) > 0 else 0.0
    print("\n=== Summary ===")
    print(f"TP={TP} FP={FP} FN={FN} TN={TN}")
    print(f"Precision={precision:.4f}")
    print(f"FPS(last avg)={fps:.2f} | Latency(last)={latency_ms:.2f} ms")
    if PSUTIL_OK:
        print(f"CPU(last)={cpu_percent:.0f}%")
    else:
        print("CPU=N/A (install psutil)")

if __name__ == "__main__":
    main()