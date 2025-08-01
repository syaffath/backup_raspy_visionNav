import time
from picamera2 import Picamera2
from yolo import YoloDetector
from vo import CameraPoses
from mobile_robot import MobileRobot
import numpy as np

# Load intrinsic matrix
with open('intrinsicNew.npy', 'rb') as f:
    intrinsic = np.load(f)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()
time.sleep(1)

yolo = YoloDetector(use_camera=False)
vo = CameraPoses(intrinsic, log_path="vo_trajectory_log.csv")
robot = MobileRobot()

prev_time = time.perf_counter()

try:
    while True:
        start_time = time.perf_counter()
        
        frame = picam2.capture_array()
        labels = yolo.get_detected_labels(frame)
        vo_pose = vo.step_with_frame(frame)
        if vo_pose is not None:
            print(f"[VO] Pose: x={vo_pose[0,3]:.2f} y={vo_pose[1,3]:.2f} z={vo_pose[2,3]:.2f}")

        bbox = yolo.get_person_bbox(frame)

        if "person" in labels and bbox is not None:
            frame_cx = frame.shape[1] // 2
            x1, y1, x2, y2 = bbox
            bbox_cx = (x1 + x2) // 2

            diff = bbox_cx - frame_cx
            margin = 40
            max_speed = 0.35
            base_speed = 0.2
            k = 0.2
            delta = k * (diff / frame_cx)
            left_speed = min(max_speed, max(0.1, base_speed - delta))
            right_speed = min(max_speed, max(0.1, base_speed + delta))

            if abs(diff) < margin:
                print("PERSON sudah di tengah, robot berhenti.")
                robot.stop()
                break
            else:
                # Belok proporsional sambil maju
                print(f"Steering: left={left_speed:.2f}, right={right_speed:.2f}, diff={diff}")
                robot.kendali_speed(left_speed, right_speed, 4)

        else:
            # Person tidak terdeteksi, boleh cari objek atau jalan pelan
            robot.kendali_speed(0.18, 0.18, 3)





        end_time = time.perf_counter()
        fps = 1.0 / (end_time - prev_time)
        print(f"[FPS] = {fps:.2f}")
        prev_time = end_time

        time.sleep(0.1)
finally:
    picam2.close()
    yolo.close()
    robot.stop()