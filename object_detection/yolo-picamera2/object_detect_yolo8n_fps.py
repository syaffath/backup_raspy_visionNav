#TEST FPS, HASIL 3 FPS
import cv2
import time
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading

# === Setup Kamera dan Model ===
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": 'RGB888', "size": (640, 480)}
))
picam2.start()
time.sleep(2)
model = YOLO("yolov8n.pt")

# === Buffer untuk plot FPS ===
max_len = 100
fps_data = deque([0]*max_len, maxlen=max_len)
frame_id_data = deque([0]*max_len, maxlen=max_len)

# === Plot FPS ===
def update_fps_plot(i):
    ax1.clear()
    ax1.plot(frame_id_data, fps_data, label='FPS', color='blue')
    ax1.set_ylim(0, max(fps_data)+1)
    ax1.set_title("Realtime FPS Monitoring")
    ax1.set_xlabel("Frame ID")
    ax1.set_ylabel("FPS")
    ax1.grid(True)
    ax1.legend()

# === Fungsi utama deteksi ===
def run_detection():
    frame_id = 0
    print("[INFO] Running detection... Tekan 'q' untuk keluar.")

    try:
        while True:
            start_time = time.time()
            frame = picam2.capture_array()

            results = model.predict(source=frame, conf=0.5, verbose=False)
            end_time = time.time()

            # Hitung FPS
            duration = end_time - start_time
            fps = 1 / duration if duration > 0 else 0
            fps_data.append(fps)
            frame_id_data.append(frame_id)

            print(f"[Frame {frame_id:04}] FPS: {fps:.2f}")

            # Tampilkan hasil deteksi
            annotated = results[0].plot()
            cv2.imshow("YOLOv8 Detection Live", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_id += 1

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt: Exiting gracefully...")

    finally:
        cv2.destroyAllWindows()
        picam2.close()
        print("[INFO] Program selesai.")

# === Setup Plot Window FPS ===
fig1, ax1 = plt.subplots()
ani1 = animation.FuncAnimation(fig1, update_fps_plot, interval=500)

# === Jalankan deteksi dan plotting FPS ===
threading.Thread(target=run_detection, daemon=True).start()
plt.show()
