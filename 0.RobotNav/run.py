import time
import cv2
from picamera2 import Picamera2
from yolo import YoloDetector
from vo import CameraPoses
from mobile_robot import MobileRobot
import numpy as np
import random
from mpu9250 import MPU9250
from logger_util import CSVLogger

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
imu = MPU9250()
prev_frame = None
frame_idx = 0

x_gt, y_gt = 0.0, 0.0
prev_total_pulse = 0
prev_yaw_deg = imu.get_orientation()[2]

yolo_logger = CSVLogger('yolo_detection_log.csv', ['label','conf','x1','y1','x2','y2'])
groundtruth_logger = CSVLogger('groundtruth_log.csv', ['frame_idx','timestamp','x_gt','y_gt', 'z_gt','yaw','pitch','roll'])

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

        # 1. Capture
        frame = picam2.capture_array()
        frame_bgr = frame
        t_capture = time.time()

        # 2. YOLO (per 6 frame)
        if frame_idx % 1 == 0:
            last_labels = yolo.get_detected_labels(frame)
            last_bbox = yolo.get_person_bbox(frame)
            last_obstacle_bbox = yolo.get_obstacle_bbox(frame)
        labels = last_labels
        bbox = last_bbox
        obstacle_bbox = last_obstacle_bbox
        t_yolo = time.time()

        # 3. Visual Odometry
        vo_pose = vo.step_with_frame(frame)
        kp = vo.orb.detect(frame_bgr, None)
        frame_orb = cv2.drawKeypoints(frame_bgr, kp, None, color=(0,255,0), flags=0)
        t_vo = time.time()

        # 4. GroundTruth
        pitch, roll, yaw = imu.get_orientation()
        total_pulse = (robot.count_left + robot.count_right) / 2
        delta_pulse = total_pulse - prev_total_pulse
        distance_cm = delta_pulse / robot.pulse_per_cm
        prev_total_pulse = total_pulse
        yaw_rad = np.deg2rad(yaw)
        x_gt += distance_cm * np.cos(yaw_rad)
        y_gt += distance_cm * np.sin(yaw_rad)
        timestamp = time.time()
        groundtruth_logger.log([frame_idx, timestamp, x_gt, y_gt, 0, yaw, pitch, roll])
        t_gt = time.time()

        frame_cx = frame.shape[1] // 2

        # 5. Control & Decision
        if is_turning:
            # Sedang belok kanan
            if abs(robot.count_left) < target_pulse and abs(robot.count_right) < target_pulse:
                print("[TURNING] Robot masih belok... kamera tetap update")
                robot.kit.motor2.throttle = -0.3
                robot.kit.motor1.throttle = 0.3
            else:
                robot.stop()
                is_turning = False
                print("[TURNING] Selesai belok kanan")
                time.sleep(2)
        elif "person" in labels and bbox is not None:
            x1, y1, x2, y2, conf = bbox
            bbox_cx = (x1 + x2) // 2
            yolo_logger.log(["person", conf, x1, y1, x2, y2])
            error = bbox_cx - frame_cx
            error_norm = error / frame_cx
            margin = 40
            K = 0.1
            forward_speed = 0.3
            turn_speed = K * error_norm
            right_speed = forward_speed - turn_speed
            left_speed = forward_speed + turn_speed
            if abs(error) < margin:
                print("OBJECT sudah di tengah, robot berhenti.")
                robot.stop()
                cv2.imshow("YOLO Live", frame_bgr)
                cv2.waitKey(10000)
                break
            else:
                print(f"Steering: left={left_speed:.2f}, right={right_speed:.2f}, diff={error}")
                robot.set_speed(left_speed, right_speed)

        elif obstacle_bbox is not None:
            label, conf, x1, y1, x2, y2 = obstacle_bbox
            yolo_logger.log([frame_idx, timestamp, label, conf, x1, y1, x2, y2])
            bbox_cx = (x1 + x2) // 2
            bbox_width = x2 - x1
            margin = 20
            if bbox_width > frame.shape[1] // 2:
                right_gap = x1
                left_gap = frame.shape[1] - x2
                target_x = x2 + margin if left_gap > right_gap else x1 - margin
                print('longgar ke kanan' if left_gap > right_gap else 'longgar ke kiri')
                error = target_x - frame_cx
            else:
                error = bbox_cx - frame_cx
                print('obstacle kecil - arah ke tengah')
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
            print(f"[Obstacle] Steering: L={left_speed:.2f} R={right_speed:.2f} width={bbox_width}")

        else:
            if prev_frame is not None and vo.is_stuck(prev_frame, frame):
                print("[VO] Robot diduga mentok/tidak bergerak, mulai belok kanan...")
                target_pulse = robot.belok_kanan_derajat(0.3, 90)
                #time.sleep(2)
                is_turning = True
            else:
                forward_speed = 0.3
                robot.set_speed(forward_speed, forward_speed)
                print(f"[NO Obstacle] Steering: L={forward_speed:.2f} R={forward_speed:.2f}")

        t_control = time.time()

        frame_idx += 1
        prev_frame = frame

        # Show
        cv2.imshow("YOLO Live", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        loop_end = time.time()

        # Timing info
        print(f"[TIMING] Capture: {(t_capture - loop_start)*1000:.1f} ms | "
              f"YOLO: {(t_yolo - t_capture)*1000:.1f} ms | "
              f"VO: {(t_vo - t_yolo)*1000:.1f} ms | "
              f"GroundTruth: {(t_gt - t_vo)*1000:.1f} ms | "
              f"Control+Show: {(loop_end - t_gt)*1000:.1f} ms | "
              f"Total: {(loop_end - loop_start)*1000:.1f} ms "
              f"({1/(loop_end - loop_start):.2f} FPS)")

finally:
    picam2.close()
    yolo.close()
    robot.stop()
    yolo_logger.close()
    groundtruth_logger.close()
    cv2.destroyAllWindows()
