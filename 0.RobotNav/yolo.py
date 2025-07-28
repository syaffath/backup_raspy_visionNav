# yolo.py
import time
from picamera2 import Picamera2
from ultralytics import YOLO

def run_yolo(shared_status=None, stop_flag=None):
    """
    YOLO loop, update shared_status['person'] setiap frame.
    """
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": 'RGB888', "size": (640, 480)}
    ))
    picam2.start()
    time.sleep(2)
    model = YOLO("yolov8n.pt")

    try:
        while True:
            frame = picam2.capture_array()
            results = model.predict(source=frame, conf=0.5, verbose=False)
            boxes = results[0].boxes
            names = results[0].names
            detected = set()
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = names[cls_id]
                if class_name == 'person':
                    detected.add('person')
            if shared_status is not None:
                shared_status['person'] = ('person' in detected)
            if stop_flag is not None and stop_flag.get('stop', False):
                print("[YOLO] Stop flag detected, exiting.")
                break
            time.sleep(0.01)
    finally:
        picam2.close()
