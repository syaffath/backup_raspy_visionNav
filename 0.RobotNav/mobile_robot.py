# mobile_robot.py
import time
import board
from adafruit_motorkit import MotorKit
from gpiozero import Button
import random

ENCODER_LEFT_PIN = 17
ENCODER_RIGHT_PIN = 22

left_encoder = Button(ENCODER_LEFT_PIN)
right_encoder = Button(ENCODER_RIGHT_PIN)
count_left = 0
count_right = 0

def inc_left():
    global count_left
    count_left += 1

def inc_right():
    global count_right
    count_right += 1

left_encoder.when_pressed = inc_left
right_encoder.when_pressed = inc_right

kit = MotorKit(i2c=board.I2C())

def stop():
    kit.motor1.throttle = 0
    kit.motor2.throttle = 0

def maju(speed=0.5, durasi=1.0):
    kit.motor1.throttle = speed
    kit.motor2.throttle = speed
    time.sleep(durasi)
    stop()

def mundur(speed=0.5, durasi=1.0):
    kit.motor1.throttle = -speed
    kit.motor2.throttle = -speed
    time.sleep(durasi)
    stop()

def belok_kiri(speed=0.5, durasi=0.5):
    kit.motor1.throttle = -speed
    kit.motor2.throttle = speed
    time.sleep(durasi)
    stop()

def belok_kanan(speed=0.5, durasi=0.5):
    kit.motor1.throttle = speed
    kit.motor2.throttle = -speed
    time.sleep(durasi)
    stop()

def robot_random_movement(shared_status=None, stop_flag=None):
    """
    Fungsi ini dijalankan di thread terpisah dari main.py
    shared_status: dict, status hasil YOLO/VO, misal {'person': False}
    stop_flag: dict, jika stop_flag['stop'] True maka robot berhenti
    """
    actions = [maju, belok_kiri, belok_kanan]
    try:
        print("[ROBOT] Random movement started.")
        while True:
            # Cek flag external (misal dari YOLO)
            if shared_status is not None and shared_status.get('person', False):
                stop()
                print("[ROBOT] Person detected! Robot STOP.")
                time.sleep(0.5)
                continue
            if stop_flag is not None and stop_flag.get('stop', False):
                print("[ROBOT] Stop flag detected, exiting robot loop.")
                break
            aksi = random.choice(actions)
            aksi()
            time.sleep(0.2)
    except Exception as e:
        stop()
        print("[ROBOT] Exception:", e)
    finally:
        stop()
        print("[ROBOT] Robot berhenti.")
