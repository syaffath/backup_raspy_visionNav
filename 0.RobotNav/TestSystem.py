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
#fps_logger = CSVLogger('fps_log.csv', ['frame_idx','timestamp','FPS'])


try:
    while True:
        #VO and YOLO
        frame = picam2.capture_array()
        frame_bgr = frame #cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        labels = yolo.get_detected_labels(frame)
        vo_pose = vo.step_with_frame(frame)
        kp = vo.orb.detect(frame_bgr, None)
        frame_orb = cv2.drawKeypoints(frame_bgr, kp, None, color=(0,255,0), flags=0)
        bbox = yolo.get_person_bbox(frame)
        frame_cx = frame.shape[1] // 2
        
        
        #GroundTruth
        pitch, roll, yaw = imu.get_orientation() #print(f"[IMU] Pitch: {pitch:.2f}, Roll: {roll:.2f}, Yaw: {yaw:.2f}")
        total_pulse = (robot.count_left + robot.count_right) / 2  # rata-rata
        delta_pulse = total_pulse - prev_total_pulse
        distance_cm = delta_pulse / robot.pulse_per_cm
        prev_total_pulse = total_pulse
        yaw_rad = np.deg2rad(yaw)
        # Update posisi (dead-reckoning)
        x_gt += distance_cm * np.cos(yaw_rad)
        y_gt += distance_cm * np.sin(yaw_rad)

        #logger
        timestamp = time.time()
        groundtruth_logger.log([frame_idx, timestamp, x_gt, y_gt, 0, yaw, pitch, roll])
        #fps_logger.writerow([frame_idx, timestamp, FPS])

        # ----- Bagian kendali robot + visualisasi (SESUAI PUNYAMU) -----
        if "person" in labels and bbox is not None:
            x1, y1, x2, y2, conf = bbox
            bbox_cx = (x1 + x2) // 2
            bbox_cy = (y1 + y2) // 2
            yolo_logger.log(["person", conf, x1, y1, x2, y2])

            # Visualisasi: Draw bbox, tengah bbox, dan garis ke tengah frame
            #cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0,255,0), 2)
            #cv2.circle(frame_bgr, (bbox_cx, bbox_cy), 5, (0,0,255), -1)
            #cv2.line(frame_bgr, (frame_cx, 0), (frame_cx, frame.shape[0]), (255,0,0), 1)
            #cv2.putText(frame_bgr, f"delta_x: {bbox_cx-frame_cx}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            error = bbox_cx - frame_cx
            error_norm = error/frame_cx
            margin = 40
            K = 0.1

            forward_speed = 0.3           # boleh juga adaptif, misal makin dekat object makin pelan
            turn_speed    = K*error_norm         # -0.18..+0.18, sesuai error

            right_speed = forward_speed - turn_speed
            left_speed  = forward_speed + turn_speed

            if abs(error) < margin:
                cv2.putText(frame_bgr, "OBJECT CENTERED - STOP!", (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                print("OBJECT sudah di tengah, robot berhenti.")
                #time.sleep(2)
                robot.stop()
                cv2.imshow("YOLO Live", frame_bgr)
                cv2.waitKey(10000)  # Tahan sejenak biar pesan terlihat
                break
            else:
                # Belok proporsional sambil maju
                print(f"Steering: left={left_speed:.2f}, right={right_speed:.2f}, diff={error}")
                robot.set_speed(left_speed, right_speed)
                #cv2.putText(frame_bgr, f"Steering: L={left_speed:.2f} R={right_speed:.2f}", 
                #            (10, frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            obstacle_bbox = yolo.get_obstacle_bbox(frame)
            if obstacle_bbox is not None:
                label, conf, x1, y1, x2, y2 = obstacle_bbox
                yolo_logger.log([frame_idx, timestamp, label, conf, x1, y1, x2, y2])
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
                #print(error_norm)
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
                
                robot.set_speed(left_speed, right_speed)    
                print(f"[Obstacle] Steering: L={left_speed:.2f} R={right_speed:.2f} width={bbox_width}")
                
                #cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0,128,255), 2)
                #cv2.putText(frame_bgr, f"Obstacle width: {bbox_width}", (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,255), 2)
                #cv2.putText(frame_bgr, f"Obstacle height: {bbox_height}", (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,255), 2)

            else:
                if prev_frame is not None:
                    if vo.is_stuck(prev_frame, frame):
                        print("[VO] Robot diduga mentok/tidak bergerak, putar 90 derajat!")
                        robot.belok_kanan_derajat(0.4, 90)
                        print('udah putar 90 derajat')    
                        time.sleep(5)
                    else:
                        forward_speed = 0.3
                        robot.set_speed(forward_speed, forward_speed)  
                        print(f"[NO Obstacle] Steering: L={forward_speed:.2f} R={forward_speed:.2f}")      
                # Tidak ada obstacle → jalan lurus

                forward_speed = 0.5
                robot.set_speed(forward_speed, forward_speed)
                print(f"[NO Obstacle] Steering: L={forward_speed:.2f} R={forward_speed:.2f}")

        frame_idx += 1
        
        #cv2.putText(frame_bgr, f"[FPS] = {1/(t_end-t0):.2f}", (frame.shape[1]-170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,128,0), 2)
        cv2.imshow("YOLO Live", frame_bgr)
        #cv2.imshow("ORB Keypoints", frame_orb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        prev_frame = frame
        

finally:
    picam2.close()
    yolo.close()
    robot.stop()
    #vo_logger.close()
    yolo_logger.close()
    groundtruth_logger.close()
    #fps_logger.close()
    cv2.destroyAllWindows()