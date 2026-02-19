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

def main():
    # LED setup (no timing, no cooldown)
    led = None
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

    # Latency metric
    latency_ms = 0.0

    # CPU metric
    cpu_percent = 0.0
    last_cpu_t = time.perf_counter()
    if PSUTIL_OK:
        psutil.cpu_percent(None)  # warm up

    # Manual labeling metrics (as you requested)
    detections_total = 0   # counts detection events (rising edge)
    TP = 0                 # only increments when you press 'p' during detection
    FP = 0                 # stays 0 because you do NOT want auto-FP
    prev_detected = False
    current_event_confirmed = False  # TP confirmed for current detection event

    print("Controls: 'm' change mode | 'p' confirm TP (only when detected) | 'q' or ESC quit")

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
    print("Press 'm' to change mode, 'p' to confirm TP, 'ESC' or 'q' to quit")

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

        # Visualization base
        out = cv2.cvtColor(vid_blur, cv2.COLOR_GRAY2BGR)

        canny_display = cv2.cvtColor(vid_canny, cv2.COLOR_GRAY2BGR)
        canny_small = cv2.resize(canny_display, (W // 4, H // 4))
        out[10:10 + H // 4, 10:10 + W // 4] = canny_small

        detected = (best_loc is not None and best_score >= thresh and (best_scale is not None and best_scale >= 0.1))

        # ---------------- Detection event counting (no auto FP) ----------------
        # Count only when detection starts (rising edge)
        if detected and not prev_detected:
            detections_total += 1
            current_event_confirmed = False  # new event, not confirmed yet

        # LED behavior: ON while detected, OFF otherwise (no seconds logic)
        if led is not None:
            if detected:
                led.on()
            else:
                led.off()

        # Draw bbox if detected
        if detected:
            x, y = best_loc
            tw, th = best_size
            cv2.rectangle(out, (x, y), (x + tw, y + th), (0, 255, 0), 2)
            cv2.putText(out, f"score={best_score:.3f} scale={best_scale:.2f}",
                        (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(out, f"no detect (best={best_score:.3f} < thr={thresh:.2f})",
                        (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ---------------- CPU (approx, update every 0.5s) ----------------
        if PSUTIL_OK and (time.perf_counter() - last_cpu_t) > 0.5:
            cpu_percent = psutil.cpu_percent(None)
            last_cpu_t = time.perf_counter()

        # ---------------- Metrics overlay ----------------
        # Unconfirmed detections = detections_total - TP (since FP stays 0 by your rule)
        unconfirmed = detections_total - TP
        # "Precision" here is "confirmed TP / total detections" (you can report it this way)
        precision = (TP / detections_total) if detections_total > 0 else 0.0

        line1 = f"Mode:{detection_mode.upper()} | FPS:{fps:.1f} | Lat:{latency_ms:.1f}ms"
        line2 = f"TP:{TP} | Detections:{detections_total} | Unconfirmed:{unconfirmed} | Prec:{precision:.3f}"
        if PSUTIL_OK:
            line3 = f"CPU:{cpu_percent:.0f}% | thr:{thresh:.2f} | Press 'p' to confirm TP"
        else:
            line3 = f"CPU:N/A (pip3 install psutil) | thr:{thresh:.2f} | Press 'p' to confirm TP"

        cv2.putText(out, line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.putText(out, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.putText(out, line3, (10, 71), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        cv2.imshow("Multi-Scale Template Matching + Canny", out)

        # ---------------- Keys ----------------
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        elif key == ord("m"):
            modes = ["gray", "canny", "hybrid"]
            current_idx = modes.index(detection_mode)
            detection_mode = modes[(current_idx + 1) % 3]
            print(f"Switched to mode: {detection_mode}")
        elif key == ord("p"):
            # Confirm TP only if we are currently detected and not confirmed yet for this event
            if detected and not current_event_confirmed:
                TP += 1
                current_event_confirmed = True
                print("TP confirmed for current detection event.")
            elif not detected:
                print("No detection right now. Put the object in front of the camera and try again.")
            else:
                print("This detection event is already confirmed as TP.")

        prev_detected = detected

    # Cleanup
    vid.release()
    cv2.destroyAllWindows()
    if led is not None:
        led.off()

    # Final summary (console)
    unconfirmed = detections_total - TP
    precision = (TP / detections_total) if detections_total > 0 else 0.0
    print("\n=== Summary ===")
    print(f"TP={TP} | Detections={detections_total} | Unconfirmed={unconfirmed}")
    print(f"Precision(TP/Detections)={precision:.4f}")
    print(f"FPS(last avg)={fps:.2f} | Latency(last)={latency_ms:.2f} ms")
    if PSUTIL_OK:
        print(f"CPU(last)={cpu_percent:.0f}%")
    else:
        print("CPU=N/A (install psutil)")

if __name__ == "__main__":
    main()