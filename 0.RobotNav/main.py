# main.py
import threading
from yolo import run_yolo
from mobile_robot import robot_random_movement
from vo import run_vo

shared_status = {'person': False}
shared_pose = {'pose': None}
stop_flag = {'stop': False}

t_yolo = threading.Thread(target=run_yolo, args=(shared_status, stop_flag))
t_robot = threading.Thread(target=robot_random_movement, args=(shared_status, stop_flag))
t_vo = threading.Thread(target=run_vo, args=(shared_pose, stop_flag))

t_yolo.start()
t_robot.start()
t_vo.start()

try:
    while True:
        if shared_status['person']:
            print("!!! PERSON DETECTED !!!")
        if shared_pose['pose'] is not None:
            print("POSE:", shared_pose['pose'])
        time.sleep(1)
except KeyboardInterrupt:
    stop_flag['stop'] = True
    print("Main: Exiting ...")
    t_yolo.join()
    t_robot.join()
    t_vo.join()
