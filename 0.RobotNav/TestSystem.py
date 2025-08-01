import time
import cv2
from picamera2 import Picamera2
from yolo import YoloDetector
from vo import CameraPoses
from mobile_robot import MobileRobot
import numpy as np
import random
from mpu9250 import MPU9250

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

try:
    while True:
        t0 = time.perf_counter()
        frame = picam2.capture_array()
        t1 = time.perf_counter()
        
        frame_bgr = frame#cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        t2 = time.perf_counter()
        
        labels = yolo.get_detected_labels(frame)
        t3 = time.perf_counter()
        
        vo_pose = vo.step_with_frame(frame)
        kp = vo.orb.detect(frame_bgr, None)
        frame_orb = cv2.drawKeypoints(frame_bgr, kp, None, color=(0,255,0), flags=0)


        #pitch, roll, yaw = imu.get_orientation()
        #print(f"[IMU] Pitch: {pitch:.2f}, Roll: {roll:.2f}, Yaw: {yaw:.2f}")
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

            error = bbox_cx - frame_cx
            error_norm = error/frame_cx
            margin = 40
            K = 0.1

            forward_speed = 0.3           # boleh juga adaptif, misal makin dekat object makin pelan
            turn_speed    = K*error_norm         # -0.18..+0.18, sesuai error

            right_speed = forward_speed - turn_speed
            left_speed  = forward_speed + turn_speed
            #print(turn_speed)

            if abs(error) < margin:
                cv2.putText(frame_bgr, "PERSON CENTERED - STOP!", (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                print("PERSON sudah di tengah, robot berhenti.")
                robot.stop()
                cv2.imshow("YOLO Live", frame_bgr)
                cv2.waitKey(500)  # Tahan sejenak biar pesan terlihat
                break
            else:
                # Belok proporsional sambil maju
                print(f"Steering: left={left_speed:.2f}, right={right_speed:.2f}, diff={error}")
                robot.set_speed(left_speed, right_speed)
                cv2.putText(frame_bgr, f"Steering: L={left_speed:.2f} R={right_speed:.2f}", 
                            (10, frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            obstacle_bbox = yolo.get_obstacle_bbox(frame)
            if obstacle_bbox is not None:
                x1, y1, x2, y2, conf = obstacle_bbox
                bbox_cx = (x1 + x2) // 2
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                frame_cx = frame.shape[1] // 2
                frame_h  = frame.shape[0]
                margin = 20  # supaya tidak terlalu mepet, bisa diubah

                if bbox_width > frame.shape[1] // 2:
                    # Obstacle sangat lebar, hindari ke sisi paling lebar
                    right_gap  = x1
                    left_gap = frame.shape[1] - x2
                    if left_gap > right_gap:
                        # Lebih longgar ke kanan, keluar ke kanan
                        target_x = x2 + margin
                        print('longgar ke kanan')
                    else:
                        # Lebih longgar ke kiri
                        target_x = x1 - margin
                        print('longgar ke kiri')
                    error = target_x - frame_cx
                else:
                    # Obstacle kecil, masih boleh arahkan ke tengah
                    error = bbox_cx - frame_cx
                    print('obstacle kecil - arah ke tengah')

                error_norm = error / frame_cx
                print(error_norm)
                # Kendali belok adaptif
                K = 0.1
                turn_speed = K * error_norm

                # Adaptasi speed maju
                forward_speed = 0.3

                max_speed = 0.5
                min_speed = -0.5

                right_speed = forward_speed - turn_speed
                left_speed  = forward_speed + turn_speed

                right_speed = max(min_speed, min(max_speed, right_speed))
                left_speed  = max(min_speed, min(max_speed, left_speed))

                print(f"[Obstacle] Steering: L={left_speed:.2f} R={right_speed:.2f} width={bbox_width}")
                robot.set_speed(left_speed, right_speed)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0,128,255), 2)
                cv2.putText(frame_bgr, f"Obstacle width: {bbox_width}", (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,255), 2)
                cv2.putText(frame_bgr, f"Obstacle height: {bbox_height}", (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,255), 2)

            else:
                if prev_frame is not None:
                    if vo.is_stuck(prev_frame, frame):
                        print("[VO] Robot diduga mentok/tidak bergerak, putar 90 derajat!")
                        robot.belok_kanan_derajat(0.3, 90)
                        print('udah putar 90 derajat')       
                # Tidak ada obstacle → jalan lurus

                forward_speed = 0.3
                cv2.putText(frame_bgr, "No person detected", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                print(f"[NO Obstacle] Steering: L={forward_speed:.2f} R={forward_speed:.2f}")
                robot.set_speed(forward_speed, forward_speed)


        t_end = time.perf_counter()
        # Print profiling timing
        print(
            f"TIMING: Capture={t1-t0:.3f}s | RGB2BGR={t2-t1:.3f}s | "
            f"YOLO_labels={t3-t2:.3f}s | VO={t4-t3:.3f}s | YOLO_bbox={t5-t4:.3f}s | "
            f"Control+Show={t_end-t5:.3f}s | Total={t_end-t0:.3f}s | FPS={1/(t_end-t0):.2f}"
        )

        cv2.putText(frame_bgr, f"[FPS] = {1/(t_end-t0):.2f}", (frame.shape[1]-170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,128,0), 2)
        cv2.imshow("YOLO Live", frame_bgr)
        cv2.imshow("ORB Keypoints", frame_orb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        prev_frame = frame
        #time.sleep(0.1)
finally:
    picam2.close()
    yolo.close()
    robot.stop()
    cv2.destroyAllWindows()
