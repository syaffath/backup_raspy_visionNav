import time
import math
import FaBo9Axis_MPU9250
from kalman_filter import KalmanFilter

class MPU9250:
    def __init__(self):
        self.imu = FaBo9Axis_MPU9250.MPU9250()
        self.kf_pitch = KalmanFilter()
        self.kf_roll = KalmanFilter()
        self.prev_time = time.time()

    def get_orientation(self):
        accel = self.imu.readAccel()
        gyro = self.imu.readGyro()
        mag = self.imu.readMagnet()

        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        ax, ay, az = accel['x'], accel['y'], accel['z']
        gx, gy, gz = gyro['x'], gyro['y'], gyro['z']
        mx, my, mz = mag['x'], mag['y'], mag['z']

        accel_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))
        accel_roll = math.degrees(math.atan2(ay, az))

        pitch = self.kf_pitch.get_angle(accel_pitch, gy, dt)
        roll = self.kf_roll.get_angle(accel_roll, gx, dt)

        roll_rad = math.radians(roll)
        pitch_rad = math.radians(pitch)
        mx2 = mx * math.cos(pitch_rad) + mz * math.sin(pitch_rad)
        my2 = mx * math.sin(roll_rad) * math.sin(pitch_rad) + my * math.cos(roll_rad) - mz * math.sin(roll_rad) * math.cos(pitch_rad)
        yaw = math.degrees(math.atan2(-my2, mx2))

        return pitch, roll, yaw
