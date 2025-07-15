import cv2
import time
import csv
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque, defaultdict
import threading

# === Setup Kamera dan Model ===
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": 'RGB888', "size": (640, 480)}
))
picam2.start()
time.sleep(2)
model = YOLO("yolov8n.pt")

# === Setup Logging ===
detection_log = open("live_detections.csv", mode='w', newline='')
fps_log = open("live_fps_log.csv", mode='w', newline='')
mean_log = open("mean_conf_log.csv", mode='w', newline='')

det_writer = csv.writer(detection_log)
fps_writer = csv.writer(fps_log)
mean_writer = csv.writer(mean_log)

det_writer.writerow(['frame_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'])
fps_writer.writerow(['frame_id', 'inference_time_s', 'fps'])
mean_writer.writerow(['frame_id', 'class_name', 'mean_conf'])

# === Buffer untuk plot ===
max_len = 100
fps_data = deque([0]*max_len, maxlen=max_len)
frame_id_data = deque([0]*max_len, maxlen=max_len)
frame_ids = deque([], maxlen=max_len)
mean_conf_history = defaultdict(lambda: deque([], maxlen=max_len))  # class_name → [mean1, mean2, ...]

# === Total mean confidence kumulatif
total_conf_history = defaultdict(list)  # class_name → list of all confidences

class_colors = ['r', 'g', 'b', 'm', 'c', 'y', 'orange', 'purple', 'brown', 'teal']

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

# === Plot Mean Confidence per Class ===
def update_mean_plot(i):
    ax2.clear()
    if len(frame_ids) == 0:
        ax2.set_title("Waiting for detection...")
        return
    
    for idx, (cls_name, history) in enumerate(mean_conf_history.items()):
        if len(history) > 0:
            # Sinkronkan panjang frame_ids dan history
            len_check = min(len(frame_ids), len(history))
            if len_check > 0:
                frame_subset = list(frame_ids)[-len_check:]
                history_subset = list(history)[-len_check:]
                ax2.plot(frame_subset, history_subset, label=f'{cls_name}', color=class_colors[idx % len(class_colors)])

    ax2.set_title("Mean Confidence per Class")
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("Mean Confidence")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True)
    ax2.legend(loc='upper right')

# === Plot Total Mean per Class ===
def update_total_mean_plot(i):
    ax3.clear()
    class_names = []
    total_means = []

    for cls_name, confs in total_conf_history.items():
        if len(confs) > 0:
            class_names.append(cls_name)
            total_means.append(np.mean(confs))

    if class_names:
        ax3.bar(class_names, total_means, color='skyblue')
        ax3.set_ylim(0, 1.05)
        ax3.set_title("Cumulative Mean Confidence per Class")
        ax3.set_ylabel("Mean Confidence")
        ax3.set_xlabel("Class Name")
        ax3.grid(axis='y')

# === Fungsi utama deteksi ===
def run_detection():
    frame_id = 0
    print("[INFO] Running detection... Tekan 'q' untuk keluar.")

    try:
        while True:
            start_time = time.time()
            frame = picam2.capture_array()
            results = model.predict(source=frame, conf=0.5, verbose=False)
            boxes = results[0].boxes
            names = results[0].names

            class_conf = defaultdict(list)

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = names[cls_id]

                det_writer.writerow([frame_id, class_name, conf, x1, y1, x2, y2])
                class_conf[class_name].append(conf)

            end_time = time.time()
            duration = end_time - start_time
            fps = 1 / duration if duration > 0 else 0
            fps_writer.writerow([frame_id, round(duration, 3), round(fps, 2)])
            fps_data.append(fps)
            frame_id_data.append(frame_id)
            frame_ids.append(frame_id)

            print(f"\n[Frame {frame_id:04}] Mean Confidence:")
            for class_name, confs in class_conf.items():
                mean_conf = np.mean(confs)
                mean_conf_history[class_name].append(mean_conf)
                mean_writer.writerow([frame_id, class_name, round(mean_conf, 3)])
                total_conf_history[class_name].extend(confs)
                print(f"{class_name}: mean={mean_conf:.3f}, n={len(confs)}")

            for class_name in mean_conf_history:
                if class_name not in class_conf:
                    mean_conf_history[class_name].append(0.0)

            annotated = results[0].plot()
            cv2.imshow("YOLOv8 Detection Live", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_id += 1

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt: Exiting gracefully...")

    finally:
        cv2.destroyAllWindows()
        detection_log.close()
        fps_log.close()
        mean_log.close()
        picam2.close()

        # Optional: simpan total mean per class
        with open("total_mean_summary.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['class_name', 'total_mean_confidence'])
            for cls, confs in total_conf_history.items():
                writer.writerow([cls, np.mean(confs)])

        print("[INFO] Logging selesai. Grafik bisa ditutup manual jika masih terbuka.")

# === Setup Plot Windows ===
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
ani1 = animation.FuncAnimation(fig1, update_fps_plot, interval=500)
ani2 = animation.FuncAnimation(fig2, update_mean_plot, interval=500)
ani3 = animation.FuncAnimation(fig3, update_total_mean_plot, interval=1000)

# === Jalankan deteksi dan plotting ===
threading.Thread(target=run_detection, daemon=True).start()
plt.show()
