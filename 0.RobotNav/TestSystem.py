import time
import cv2
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

try:
    while True:
        t0 = time.perf_counter()
        frame = picam2.capture_array()
        t1 = time.perf_counter()
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        t2 = time.perf_counter()
        
        labels = yolo.get_detected_labels(frame)
        t3 = time.perf_counter()
        
        vo_pose = vo.step_with_frame(frame)
        t4 = time.perf_counter()
        
        bbox = yolo.get_person_bbox(frame)
        t5 = time.perf_counter()
        
        frame_cx = frame.shape[1] // 2

        # ----- Bagian kendali robot + visualisasi (SESUAI PUNYAMU) -----
        if "person" in labels and bbox is not None:
            x1, y1, x2, y2, conf = bbox
            bbox_cx = (x1 + x2) // 2
            bbox_cy = (y1 + y2) // 2

            # Visualisasi: Draw bbox, tengah bbox, dan garis ke tengah frame
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame_bgr, (bbox_cx, bbox_cy), 5, (0,0,255), -1)
            cv2.line(frame_bgr, (frame_cx, 0), (frame_cx, frame.shape[0]), (255,0,0), 1)
            cv2.putText(frame_bgr, f"delta_x: {bbox_cx-frame_cx}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            diff = bbox_cx - frame_cx
            margin = 40
            max_speed = 0.3
            base_speed = 0.2
            k = 0.2
            delta = k * (diff / frame_cx)
            min_speed = 0 #0.27  # ganti dari 0.1 ke 0.15 atau 0.18
            right_speed = min(max_speed, max(min_speed, base_speed - delta))
            left_speed  = min(max_speed, max(min_speed, base_speed + delta))
            print(right_speed, left_speed)

            if abs(diff) < margin:
                cv2.putText(frame_bgr, "PERSON CENTERED - STOP!", (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                print("PERSON sudah di tengah, robot berhenti.")
                robot.stop()
                cv2.imshow("YOLO Live", frame_bgr)
                cv2.waitKey(500)  # Tahan sejenak biar pesan terlihat
                break
            else:
                # Belok proporsional sambil maju
                print(f"Steering: left={left_speed:.2f}, right={right_speed:.2f}, diff={diff}")
                robot.set_speed(left_speed, right_speed)
                cv2.putText(frame_bgr, f"Steering: L={left_speed:.2f} R={right_speed:.2f}", 
                            (10, frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            cv2.putText(frame_bgr, "No person detected", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            robot.set_speed(0.18, 0.18)

        t_end = time.perf_counter()
        # Print profiling timing
        print(
            f"TIMING: Capture={t1-t0:.3f}s | RGB2BGR={t2-t1:.3f}s | "
            f"YOLO_labels={t3-t2:.3f}s | VO={t4-t3:.3f}s | YOLO_bbox={t5-t4:.3f}s | "
            f"Control+Show={t_end-t5:.3f}s | Total={t_end-t0:.3f}s | FPS={1/(t_end-t0):.2f}"
        )

        cv2.putText(frame_bgr, f"[FPS] = {1/(t_end-t0):.2f}", (frame.shape[1]-170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,128,0), 2)
        cv2.imshow("YOLO Live", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)
finally:
    picam2.close()
    yolo.close()
    robot.stop()
    cv2.destroyAllWindows()
