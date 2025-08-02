import time
import math
import FaBo9Axis_MPU9250
from kalman_filter import KalmanFilter
from collections import deque

class MPU9250:
    def __init__(self):
        self.imu = FaBo9Axis_MPU9250.MPU9250()
        self.kf_pitch = KalmanFilter()
        self.kf_roll = KalmanFilter()
        self.prev_time = time.time()
        
        self.mag_yaw_buffer = deque(maxlen=5)  # buffer smoothing yaw magneto
        self.yaw_gyro = None  # akan di-set di loop pertama
        self.kf_yaw = KalmanFilter()

    def get_orientation(self):
        accel = self.imu.readAccel()
        gyro = self.imu.readGyro()

        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        ax, ay, az = accel['x'], accel['y'], accel['z']
        gx, gy, gz = gyro['x'], gyro['y'], gyro['z']

        # Kalman untuk pitch dan roll
        accel_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))
        accel_roll = math.degrees(math.atan2(ay, az))

        pitch = self.kf_pitch.get_angle(accel_pitch, gy, dt)
        roll = self.kf_roll.get_angle(accel_roll, gx, dt)

        # Gyro yaw
        if self.yaw_gyro is None:
            self.yaw_gyro = 0.0
        self.yaw_gyro += gz * dt

        # Wrap ke [-180, +180)
        yaw = (self.yaw_gyro + 180) % 360 - 180

        return pitch, roll, yaw
