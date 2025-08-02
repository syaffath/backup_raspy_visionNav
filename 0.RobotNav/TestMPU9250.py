import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from mpu9250 import MPU9250
import time

# Inisialisasi IMU
imu = MPU9250()

# Buffer data
window = 100
pitch_hist = deque(maxlen=window)
roll_hist = deque(maxlen=window)
yaw_hist = deque(maxlen=window)
t_hist = deque(maxlen=window)

# Setup plot
fig, ax = plt.subplots()
line_pitch, = ax.plot([], [], label='Pitch')
line_roll, = ax.plot([], [], label='Roll')
line_yaw, = ax.plot([], [], label='Yaw')

ax.set_xlim(0, window)
ax.set_ylim(-180, 180)
ax.set_title("Real-Time IMU Orientation (Kalman Filtered)")
ax.set_xlabel("Frame")
ax.set_ylabel("Angle (°)")
ax.grid(True)
ax.legend()

frame_counter = [0]  # pakai list agar bisa mutable dalam closure

def update(frame):
    pitch, roll, yaw = imu.get_orientation()
    pitch_hist.append(pitch)
    roll_hist.append(roll)
    yaw_hist.append(yaw)
    t_hist.append(frame_counter[0])

    line_pitch.set_data(range(len(pitch_hist)), pitch_hist)
    line_roll.set_data(range(len(roll_hist)), roll_hist)
    line_yaw.set_data(range(len(yaw_hist)), yaw_hist)

    frame_counter[0] += 1
    return line_pitch, line_roll, line_yaw

ani = FuncAnimation(fig, update, interval=100)
plt.tight_layout()
plt.show()
