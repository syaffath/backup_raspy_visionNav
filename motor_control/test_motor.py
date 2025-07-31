import time
import board
from adafruit_motorkit import MotorKit
from gpiozero import Button

# Pin encoder
ENCODER_LEFT_PIN = 17
ENCODER_RIGHT_PIN = 22

# Inisialisasi encoder sebagai tombol pulse
left_encoder = Button(ENCODER_LEFT_PIN)
right_encoder = Button(ENCODER_RIGHT_PIN)

# Global pulse counters
count_left = 0
count_right = 0

def inc_left():
    global count_left
    count_left += 1

def inc_right():
    global count_right
    count_right += 1

# Register callback pada event tombol encoder ditekan
left_encoder.when_pressed = inc_left
right_encoder.when_pressed = inc_right

kit = MotorKit(i2c=board.I2C())

def stop():
    kit.motor1.throttle = 0 #motor kiri
    kit.motor2.throttle = 0 #motor kanan

def maju(speed, durasi):
    kit.motor1.throttle = speed
    kit.motor2.throttle = speed
    time.sleep(durasi)
    stop()

def mundur(speed, durasi):
    kit.motor1.throttle = -speed
    kit.motor2.throttle = -speed
    time.sleep(durasi)
    stop()

def belok_kiri(speed, durasi):
    # Kanan maju, kiri mundur = berputar kiri
    kit.motor1.throttle = -speed
    kit.motor2.throttle = speed
    time.sleep(durasi)
    stop()

def belok_kanan(speed, durasi):
    # Kiri maju, kanan mundur = berputar kanan
    kit.motor1.throttle = speed
    kit.motor2.throttle = -speed
    time.sleep(durasi)
    stop()

def print_encoder():
    print(f"Pulse Kiri: {count_left}, Kanan: {count_right}")

def reset_encoder():
    global count_left, count_right
    count_left = 0
    count_right = 0

try:
    while True:
        print("\n--- MENU ---")
        print("1. Maju")
        print("2. Mundur")
        print("3. Belok Kiri")
        print("4. Belok Kanan")
        print("5. Print Pulse Encoder")
        print("6. Reset Encoder")
        print("0. Exit")
        cmd = input("Pilih: ")
        
        if cmd == "1":
            maju(0.3, 1.0)
            print_encoder()
        elif cmd == "2":
            mundur(0.3, 1.0)
            print_encoder()
        elif cmd == "3":
            belok_kiri(0.3, 0.5)
            print_encoder()
        elif cmd == "4":
            belok_kanan(0.3, 0.5)
            print_encoder()
        elif cmd == "5":
            print_encoder()
        elif cmd == "6":
            reset_encoder()
            print("Encoder direset.")
        elif cmd == "0":
            break
        else:
            print("Pilihan tidak valid.")

finally:
    stop()
    print("Robot berhenti.")
