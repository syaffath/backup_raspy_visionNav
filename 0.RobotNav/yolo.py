# yolo.py
import time
from picamera2 import Picamera2
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, 
                 model_path="yolov8n.pt", 
                 confidence=0.5,
                 use_camera=True,           # Bisa: True (bawa kamera sendiri), False (dapat frame dari main)
                 camera_format="RGB888", 
                 camera_size=(640, 480)):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.use_camera = use_camera
        if self.use_camera:
            self.picam2 = Picamera2()
            self.picam2.configure(
                self.picam2.create_preview_configuration(
                    main={"format": camera_format, "size": camera_size}
                )
            )
            self.picam2.start()
            time.sleep(1)  # warm-up kamera

    def _detect_class(self, class_name, frame=None):
        # Jika use_camera, capture frame dari kamera. Jika tidak, frame HARUS dikasih dari luar.
        if self.use_camera:
            frame = self.picam2.capture_array()
        elif frame is None:
            raise ValueError("Frame harus diberikan jika use_camera=False")
        results = self.model.predict(source=frame, conf=self.confidence, verbose=False)
        boxes = results[0].boxes
        names = results[0].names
        for box in boxes:
            cls_id = int(box.cls[0])
            nama = names[cls_id]
            if nama == class_name:
                return True
        return False

    def detect_person(self, frame=None):
        return self._detect_class("person", frame)
    """
    def get_person_bbox(self, frame):
        results = self.model.predict(source=frame, conf=self.confidence, verbose=False)
        boxes = results[0].boxes
        names = results[0].names
        for box in boxes:
            cls_id = int(box.cls[0])
            nama = names[cls_id]
            if nama == "person":
                x1, y1, x2, y2 = box.xyxy[0]  # bounding box [xmin, ymin, xmax, ymax]
                return int(x1), int(y1), int(x2), int(y2)
        return None
    """
    def get_person_bbox(self, frame):
        results = self.model.predict(source=frame, conf=self.confidence, verbose=False)
        boxes = results[0].boxes
        names = results[0].names
        max_area = 0
        bbox_result = None
        for box in boxes:
            cls_id = int(box.cls[0])
            nama = names[cls_id]
            if nama == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    bbox_result = (x1, y1, x2, y2, float(box.conf[0]))
        return bbox_result  # (x1, y1, x2, y2, conf) or None    
       
    def get_detected_labels(self, frame=None):
        """
        Deteksi semua objek dalam frame. 
        Jika use_camera=True, otomatis ambil dari kamera.
        Jika use_camera=False, HARUS passing frame=...
        """
        if self.use_camera:
            frame = self.picam2.capture_array()
        elif frame is None:
            raise ValueError("Frame harus diberikan jika use_camera=False")
        results = self.model.predict(source=frame, conf=self.confidence, verbose=False)
        boxes = results[0].boxes
        names = results[0].names
        labels = set()
        for box in boxes:
            cls_id = int(box.cls[0])
            nama = names[cls_id]
            labels.add(nama)
        return labels

    def close(self):
        if self.use_camera:
            self.picam2.close()
