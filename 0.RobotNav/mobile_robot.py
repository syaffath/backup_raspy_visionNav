import time
import board
from adafruit_motorkit import MotorKit
from gpiozero import Button

class MobileRobot:
    def __init__(self, 
                 encoder_left_pin=17, 
                 encoder_right_pin=22,
                 pulse_per_rev=20,     # ganti sesuai spesifikasi encoder
                 roda_diameter_cm=5.98 
                 ):
        self.kit = MotorKit(i2c=board.I2C())
        self.left_encoder = Button(encoder_left_pin)
        self.right_encoder = Button(encoder_right_pin)
        self.count_left = 0
        self.count_right = 0

        # Encoder setup
        self.left_encoder.when_pressed = self.inc_left
        self.right_encoder.when_pressed = self.inc_right

        # Kalkulasi pulse per cm
        self.pulse_per_rev = pulse_per_rev
        self.roda_diameter_cm = roda_diameter_cm
        self.keliling_cm = 3.1416 * roda_diameter_cm
        self.pulse_per_cm = self.pulse_per_rev / self.keliling_cm

    def inc_left(self):
        self.count_left += 1

    def inc_right(self):
        self.count_right += 1

    def reset_encoder(self):
        self.count_left = 0
        self.count_right = 0

    def stop(self):
        self.kit.motor1.throttle = 0
        self.kit.motor2.throttle = 0

    #def print_encoder(self):
        #print(f"Pulse Kiri: {self.count_left}, Kanan: {self.count_right}")

    def maju_cm(self, speed, jarak_cm):
        self.reset_encoder()
        target_pulse = int(jarak_cm * self.pulse_per_cm)
        self.kit.motor2.throttle = speed
        self.kit.motor1.throttle = speed
        while self.count_left < target_pulse and self.count_right < target_pulse:
            # Koreksi jika salah satu roda lebih cepat
            if abs(self.count_left - self.count_right) > 2:
                if self.count_left > self.count_right:
                    self.kit.motor2.throttle = speed * 0.95
                    self.kit.motor1.throttle = speed * 1.05
                else:
                    self.kit.motor2.throttle = speed * 1.05
                    self.kit.motor1.throttle = speed * 0.95
            else:
                self.kit.motor2.throttle = speed
                self.kit.motor1.throttle = speed
            #print(f"Left: {self.count_left}, Right: {self.count_right}")    
            time.sleep(0.01)
        self.stop()

    def mundur_cm(self, speed, jarak_cm):
        self.reset_encoder()
        target_pulse = int(jarak_cm * self.pulse_per_cm)
        self.kit.motor2.throttle = -speed
        self.kit.motor1.throttle = -speed
        while self.count_left < target_pulse and self.count_right < target_pulse:
            # Koreksi jika salah satu roda lebih cepat
            if abs(self.count_left - self.count_right) > 2:
                if self.count_left > self.count_right:
                    self.kit.motor2.throttle = -speed * 0.95
                    self.kit.motor1.throttle = -speed * 1.05
                else:
                    self.kit.motor2.throttle = -speed * 1.05
                    self.kit.motor1.throttle = -speed * 0.95
            else:
                self.kit.motor2.throttle = -speed
                self.kit.motor1.throttle = -speed
            #print(f"Left: {self.count_left}, Right: {self.count_right}")
            time.sleep(0.01)
        self.stop()

    def belok_kiri_derajat(self, speed, derajat, jarak_sumbu_roda_cm=10):
        # Belok kiri: roda kanan maju, roda kiri mundur
        self.reset_encoder()
        # Hitung jarak lengkung roda (arc): keliling setengah lingkaran (C=pi*D), D=jarak antar roda
        arc = (3.1416 * jarak_sumbu_roda_cm) * (derajat / 360)
        target_pulse = int(arc * self.pulse_per_cm)
        self.kit.motor2.throttle = speed
        self.kit.motor1.throttle = -speed
        while self.count_left < target_pulse and self.count_right < target_pulse:
            #print(f"Left: {self.count_left}, Right: {self.count_right}")
            time.sleep(0.01)
        self.stop()

    def belok_kanan_derajat(self, speed, derajat, jarak_sumbu_roda_cm=10):
        # Belok kanan: roda kiri maju, roda kanan mundur
        self.reset_encoder()
        arc = (3.1416 * jarak_sumbu_roda_cm) * (derajat / 360)
        target_pulse = int(arc * self.pulse_per_cm)
        self.kit.motor2.throttle = -speed
        self.kit.motor1.throttle = speed
        while self.count_left < target_pulse and self.count_right < target_pulse:
            #print(f"Left: {self.count_left}, Right: {self.count_right}")
            time.sleep(0.01)
        self.stop()

    def kendali_speed(self, left_speed, right_speed, durasi_detik):
        """
        Kendalikan motor kiri & kanan dengan speed berbeda selama durasi_detik.
        left_speed/right_speed: -1.0 ... 1.0
        durasi_detik: waktu gerak (float, detik)
        """
        self.kit.motor1.throttle = left_speed   # kiri
        self.kit.motor2.throttle = right_speed  # kanan
        time.sleep(durasi_detik)
        self.stop()

    def set_speed(self, left, right):
        self.kit.motor1.throttle = left
        self.kit.motor2.throttle = right