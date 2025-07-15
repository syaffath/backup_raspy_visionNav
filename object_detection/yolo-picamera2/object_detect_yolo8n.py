from picamera2 import Picamera2
import cv2
import time
from ultralytics import YOLO

# Inisialisasi Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": 'RGB888', "size": (640, 480)}
))
picam2.start()
time.sleep(2)

# Load model YOLOv8n (otomatis dari cache jika sudah di-download)
model = YOLO("yolov8n.pt")

while True:
    frame = picam2.capture_array()
    results = model.predict(source=frame, conf=0.5, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
