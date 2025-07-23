import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import FaBo9Axis_MPU9250

# Inisialisasi sensor
mpu9250 = FaBo9Axis_MPU9250.MPU9250()
dt = 0.1  # 100 ms sampling

# List penampung data
accel_x, accel_y, accel_z = [], [], []
gyro_x, gyro_y, gyro_z = [], [], []
mag_x, mag_y, mag_z = [], [], []

# Siapkan file CSV dan header
csv_filename = "mpu9250_data.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "timestamp",
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "mag_x", "mag_y", "mag_z"
    ])

# Setup plotting
plt.ion()
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Accel
line_ax, = axs[0].plot([], [], 'b-', label='Accel X')
line_ay, = axs[0].plot([], [], 'r-', label='Accel Y')
line_az, = axs[0].plot([], [], 'g-', label='Accel Z')
axs[0].set_ylabel('Accel (g)')
axs[0].set_title('Realtime Accelerometer (MPU9250)')
axs[0].legend()
axs[0].grid()

# Gyro
line_gx, = axs[1].plot([], [], 'b-', label='Gyro X')
line_gy, = axs[1].plot([], [], 'r-', label='Gyro Y')
line_gz, = axs[1].plot([], [], 'g-', label='Gyro Z')
axs[1].set_ylabel('Gyro (deg/s)')
axs[1].set_title('Realtime Gyroscope (MPU9250)')
axs[1].legend()
axs[1].grid()

# Magnet
line_mx, = axs[2].plot([], [], 'b-', label='Mag X')
line_my, = axs[2].plot([], [], 'r-', label='Mag Y')
line_mz, = axs[2].plot([], [], 'g-', label='Mag Z')
axs[2].set_xlabel('Step')
axs[2].set_ylabel('Mag (uT)')
axs[2].set_title('Realtime Magnetometer (MPU9250)')
axs[2].legend()
axs[2].grid()

plt.tight_layout()

def update(frame):
    ts = time.time()  # Waktu pengambilan data

    # ACCEL
    a = mpu9250.readAccel()
    accel_x.append(a['x'])
    accel_y.append(a['y'])
    accel_z.append(a['z'])

    # GYRO
    g = mpu9250.readGyro()
    gyro_x.append(g['x'])
    gyro_y.append(g['y'])
    gyro_z.append(g['z'])

    # MAG
    m = mpu9250.readMagnet()
    mag_x.append(m['x'])
    mag_y.append(m['y'])
    mag_z.append(m['z'])

    # Simpan ke CSV
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            ts,
            a['x'], a['y'], a['z'],
            g['x'], g['y'], g['z'],
            m['x'], m['y'], m['z']
        ])

    step = np.arange(1, len(accel_x) + 1)

    # Update plot Accel
    line_ax.set_data(step, accel_x)
    line_ay.set_data(step, accel_y)
    line_az.set_data(step, accel_z)
    axs[0].set_xlim(max(0, len(accel_x)-100), len(accel_x)+10)
    axs[0].set_ylim(-1.5, 1.5)  # Atur sesuai range sensor kamu

    # Update plot Gyro
    line_gx.set_data(step, gyro_x)
    line_gy.set_data(step, gyro_y)
    line_gz.set_data(step, gyro_z)
    axs[1].set_xlim(max(0, len(gyro_x)-100), len(gyro_x)+10)
    axs[1].set_ylim(-1, 1)  # Atur sesuai range sensor kamu

    # Update plot Magnet
    line_mx.set_data(step, mag_x)
    line_my.set_data(step, mag_y)
    line_mz.set_data(step, mag_z)
    axs[2].set_xlim(max(0, len(mag_x)-100), len(mag_x)+10)
    axs[2].set_ylim(-510, 105)  # Atur sesuai range sensor kamu

    # Print ke terminal (optional)
    print(f"Step: {len(accel_x)} | "
          f"Accel: ({a['x']:.2f},{a['y']:.2f},{a['z']:.2f}) | "
          f"Gyro: ({g['x']:.2f},{g['y']:.2f},{g['z']:.2f}) | "
          f"Mag: ({m['x']:.2f},{m['y']:.2f},{m['z']:.2f})")

    return (line_ax, line_ay, line_az,
            line_gx, line_gy, line_gz,
            line_mx, line_my, line_mz)

ani = FuncAnimation(fig, update, interval=int(dt*1000), blit=False)

try:
    print("Monitoring dimulai! Tekan Ctrl+C untuk berhenti.")
    plt.show(block=True)
except KeyboardInterrupt:
    print("Monitoring dihentikan.")
    plt.close()
