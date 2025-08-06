import time
import cv2
from picamera2 import Picamera2
from yolo import YoloDetector
from vo import CameraPoses
from mobile_robot import MobileRobot
import numpy as np
from mpu9250 import MPU9250
from logger_util import CSVLogger

# --- Logger untuk latency dan FPS ---
latency_logger = CSVLogger("latency_fps_log_without YOLO.csv", [
    "frame_idx", "t_capture_ms", "t_vo_ms", "t_gt_ms", "t_control_show_ms", "t_total_ms", "fps"
])

# Simpan statistik untuk rata-rata
fps_list = []
latency_list = []

# Load intrinsic matrix
with open('intrinsicNew.npy', 'rb') as f:
    intrinsic = np.load(f)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()
time.sleep(1)

yolo = YoloDetector(use_camera=False)
vo = CameraPoses(intrinsic)
robot = MobileRobot()
imu = MPU9250()
prev_frame = None
frame_idx = 0

x_gt, y_gt = 0.0, 0.0
prev_total_pulse = 0

# YOLO buffer
last_labels = []
last_bbox = None
last_obstacle_bbox = None

# Turning state
is_turning = False
target_pulse = 0

try:
    while True:
        loop_start = time.time()

        # --- 1. Capture ---
        frame = picam2.capture_array()
        frame_bgr = frame.copy()
        t_capture = time.time()
        frame_cx = frame.shape[1] // 2

        # --- 2. YOLO ---
        last_labels = yolo.get_detected_labels(frame)
        last_bbox = yolo.get_person_bbox(frame)
        last_obstacle_bbox = yolo.get_obstacle_bbox(frame)
        labels = last_labels
        bbox = last_bbox
        obstacle_bbox = last_obstacle_bbox
        t_yolo = time.time()

        # --- 3. GroundTruth ---
        pitch, roll, yaw = imu.get_orientation()
        total_pulse = (robot.count_left + robot.count_right) / 2
        delta_pulse = total_pulse - prev_total_pulse
        if delta_pulse < 0 or not np.isfinite(delta_pulse):
            delta_pulse = 0
        distance_cm = delta_pulse / robot.pulse_per_cm
        prev_total_pulse = total_pulse
        yaw_rad = np.deg2rad(yaw)
        x_gt += distance_cm * np.cos(yaw_rad)
        y_gt += distance_cm * np.sin(yaw_rad)
        t_gt = time.time()

        # --- 4. VO ---
        vo_pose, vo_log = vo.step_with_frame(frame)
        kp = vo.orb.detect(frame_bgr, None)
        frame_orb = cv2.drawKeypoints(frame_bgr, kp, None, color=(0, 255, 0), flags=0)
        frame_id, x_vo, y_vo, z_vo, yaw_vo, pitch_vo, roll_vo = vo_log
        t_vo = time.time()

        # --- 5. Control & Decision ---
        left_speed, right_speed = 0.0, 0.0

        if is_turning:
            if abs(robot.count_left) < target_pulse and abs(robot.count_right) < target_pulse:
                robot.kit.motor2.throttle = -0.3
                robot.kit.motor1.throttle = 0.3
            else:
                robot.stop()
                is_turning = False
                time.sleep(2)

        elif "person" in labels and bbox is not None:
            x1, y1, x2, y2, conf = bbox
            bbox_cx = (x1 + x2) // 2
            error = bbox_cx - frame_cx
            error_norm = error / frame_cx
            margin = 40
            K = 0.1
            forward_speed = 0.3
            turn_speed = K * error_norm
            right_speed = forward_speed - turn_speed
            left_speed = forward_speed + turn_speed
            if abs(error) < margin:
                robot.stop()
                print("ROBOT STOP!!!")
                cv2.imshow("Live Object Detection", frame_bgr)
                cv2.waitKey(100)
                break
            else:
                robot.set_speed(left_speed, right_speed)

        elif obstacle_bbox is not None:
            label, conf, x1, y1, x2, y2 = obstacle_bbox
            bbox_cx = (x1 + x2) // 2
            bbox_width = x2 - x1
            margin = 20
            if bbox_width > frame.shape[1] // 2:
                right_gap = x1
                left_gap = frame.shape[1] - x2
                target_x = x2 + margin if left_gap > right_gap else x1 - margin
                error = target_x - frame_cx
            else:
                error = bbox_cx - frame_cx
            error_norm = error / frame_cx
            K = 0.1
            turn_speed = K * error_norm
            forward_speed = 0.3
            max_speed = 0.3
            min_speed = -0.3
            right_speed = forward_speed - turn_speed
            left_speed = forward_speed + turn_speed
            right_speed = max(min_speed, min(max_speed, right_speed))
            left_speed = max(min_speed, min(max_speed, left_speed))
            robot.set_speed(left_speed, right_speed)

        else:
            if prev_frame is not None and vo.is_stuck(prev_frame, frame):
                target_pulse = robot.belok_kanan_derajat(0.3, 90)
                is_turning = True
            else:
                forward_speed = 0.3
                left_speed, right_speed = forward_speed, forward_speed
                robot.set_speed(forward_speed, forward_speed)

        t_control = time.time()

        # --- Gambar BBOX & Info ---
        if bbox is not None:
            x1, y1, x2, y2, conf = bbox
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            bbox_cx, bbox_cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame_bgr, (bbox_cx, bbox_cy), 6, (0, 0, 255), -1)
            delta_x = bbox_cx - frame_cx
            cv2.putText(frame_bgr, f"delta_x: {delta_x}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            margin = 40
            cv2.line(frame_bgr, (frame_cx - margin, 0), (frame_cx - margin, frame.shape[0]), (255, 0, 0), 2)
            cv2.line(frame_bgr, (frame_cx + margin, 0), (frame_cx + margin, frame.shape[0]), (255, 0, 0), 2)

        # FPS di pojok kanan atas
        loop_end = time.time()
        elapsed_time = loop_end - loop_start
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(frame_bgr, f"[FPS] = {fps:.2f}",
                    (frame_bgr.shape[1] - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Steering info di pojok kiri bawah
        cv2.putText(frame_bgr, f"Steering: L={left_speed:.2f} R={right_speed:.2f}",
                    (10, frame_bgr.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Latency ---
        t_capture_ms = (t_capture - loop_start) * 1000
        t_yolo_ms = (t_yolo - t_capture) * 1000
        t_gt_ms = (t_gt - t_capture) * 1000
        t_vo_ms = (t_vo - t_gt) * 1000
        t_control_show_ms = (loop_end - t_vo) * 1000
        t_total_ms = elapsed_time * 1000

        latency_logger.log([
            frame_idx, t_capture_ms, t_vo_ms, t_gt_ms, t_control_show_ms, t_total_ms, fps
        ])

        fps_list.append(fps)
        latency_list.append(t_total_ms)

        # --- Show ---
        cv2.imshow("Object Detection with BBox", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1
        prev_frame = frame

finally:
    picam2.close()
    yolo.close()
    robot.stop()
    cv2.destroyAllWindows()
    if fps_list:
        avg_fps = sum(fps_list) / len(fps_list)
        avg_latency = sum(latency_list) / len(latency_list)
        print(f"\n[STATS] Average FPS: {avg_fps:.2f}")
        print(f"[STATS] Average Latency: {avg_latency:.1f} ms")
