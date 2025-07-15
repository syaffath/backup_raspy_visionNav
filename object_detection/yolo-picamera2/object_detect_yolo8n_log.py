from picamera2 import Picamera2
import cv2
import time
from ultralytics import YOLO
import csv
import pandas as pd
import matplotlib.pyplot as plt

# Inisialisasi Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": 'RGB888', "size": (640, 480)}
))
picam2.start()
time.sleep(2)

# Load model YOLOv8n
model = YOLO("yolov8n.pt")

# Siapkan file logging
detection_log = open("detections.csv", mode='w', newline='')
fps_log = open("fps_log.csv", mode='w', newline='')
det_writer = csv.writer(detection_log)
fps_writer = csv.writer(fps_log)

det_writer.writerow(['frame_id', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2'])
fps_writer.writerow(['frame_id', 'inference_time_ms', 'fps'])

frame_id = 0

print("[INFO] Running detection... Tekan 'q' untuk keluar.")

try:
    while True:
        start_time = time.time()

        frame = picam2.capture_array()
        results = model.predict(source=frame, conf=0.5, verbose=False)

        # Ambil hasil deteksi
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_writer.writerow([frame_id, cls_id, conf, x1, y1, x2, y2])

        # Hitung FPS
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # ms
        fps = 1000 / inference_time if inference_time > 0 else 0
        fps_writer.writerow([frame_id, round(inference_time, 2), round(fps, 2)])

        # Tampilkan hasil
        annotated = results[0].plot()
        cv2.imshow("YOLOv8 Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

except KeyboardInterrupt:
    print("[INFO] KeyboardInterrupt: Exiting gracefully...")

finally:
    # Tutup semua
    cv2.destroyAllWindows()
    detection_log.close()
    fps_log.close()
    picam2.close()
    print("[INFO] Logging selesai. Menampilkan visualisasi...")

    # === VISUALISASI HASIL ===
    df_fps = pd.read_csv('fps_log.csv')
    df_det = pd.read_csv('detections.csv')

    # Plot FPS dan waktu inferensi
    plt.figure()
    plt.plot(df_fps['frame_id'], df_fps['fps'], label='FPS')
    plt.plot(df_fps['frame_id'], df_fps['inference_time_ms'], label='Inference Time (ms)')
    plt.xlabel("Frame ID")
    plt.ylabel("FPS / Time")
    plt.title("YOLOv8n Performance (Raspberry Pi 5)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot jumlah deteksi per frame
    count_per_frame = df_det.groupby('frame_id').size()
    plt.figure()
    count_per_frame.plot(kind='bar', figsize=(10, 4))
    plt.xlabel("Frame ID")
    plt.ylabel("Jumlah Deteksi")
    plt.title("Jumlah Objek Terdeteksi per Frame")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Statistik Ringkas
    print("\n== FPS Statistics ==")
    print("Rata-rata FPS:", round(df_fps['fps'].mean(), 2))
    print("Min FPS:", round(df_fps['fps'].min(), 2))
    print("Max FPS:", round(df_fps['fps'].max(), 2))

    print("\n== Deteksi ==")
    print("Total Frame dengan Deteksi:", count_per_frame.count())
    print("Total Bounding Boxes:", len(df_det))
    print("Kelas Terbanyak:", df_det['class_id'].value_counts().idxmax())
