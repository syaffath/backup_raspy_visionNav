import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import FaBo9Axis_MPU9250

# Setup sensor
mpu9250 = FaBo9Axis_MPU9250.MPU9250()

dt = 0.1   # 100ms sampling

# ==== Zero Velocity Update Settings ====
zvu_accel_thresh = 0.12    # m/s^2, threshold akselerasi (absolut)
zvu_step_hold = 5         # harus diam selama N step sebelum velocity benar-benar direset

# Data X & Y
accel_x_list = []
velocity_x_list = [0.0]
position_x_list = [0.0]
zvu_counter_x = 0

accel_y_list = []
velocity_y_list = [0.0]
position_y_list = [0.0]
zvu_counter_y = 0

# Plot setup (gabungkan velocity & posisi X dan Y)
plt.ion()
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
line_vx, = axs[0].plot([], [], 'b-', label='Velocity X (m/s)')
line_vy, = axs[0].plot([], [], 'g-', label='Velocity Y (m/s)')
line_px, = axs[1].plot([], [], 'r-', label='Position X (m)')
line_py, = axs[1].plot([], [], 'm-', label='Position Y (m)')
axs[0].set_ylabel("Velocity (m/s)")
axs[1].set_ylabel("Position (m)")
axs[1].set_xlabel("Step")
axs[0].grid(); axs[1].grid()
axs[0].legend(); axs[1].legend()
plt.tight_layout()

def update(frame):
    global zvu_counter_x, zvu_counter_y

    # Ambil data akselerasi dari sensor
    a = mpu9250.readAccel()
    a['x'] = round(a['x'], 2)
    a['y'] = round(a['y'], 2)

    if abs(a['x']) < 0.11:
        a['x'] = 0    
    #if abs(a['y']) < 0.11:
    #    a['y'] = 0

    ax_ms2 = a['x']  # asumsikan sudah dalam m/s^2
    ay_ms2 = a['y']

    accel_x_list.append(ax_ms2)
    accel_y_list.append(ay_ms2)

    # ==== Zero Velocity Update X ====
    if abs(ax_ms2) < zvu_accel_thresh:
        zvu_counter_x += 1
    else:
        zvu_counter_x = 0

    # ==== Zero Velocity Update Y ====
    if abs(ay_ms2) < zvu_accel_thresh:
        zvu_counter_y += 1
    else:
        zvu_counter_y = 0

    # --- Integrasi X (trapezoidal) ---
    if len(accel_x_list) == 1:
        v_x_new = 0.0
    else:
        v_x_new = velocity_x_list[-1] + ((accel_x_list[-2] + accel_x_list[-1]) / 2) * dt
    # ZVU untuk X
    if zvu_counter_x >= zvu_step_hold:
        v_x_new = 0.0
    velocity_x_list.append(v_x_new)
    x_new = position_x_list[-1] + v_x_new * dt
    position_x_list.append(x_new)

    # --- Integrasi Y (trapezoidal) ---
    if len(accel_y_list) == 1:
        v_y_new = 0.0
    else:
        v_y_new = velocity_y_list[-1] + ((accel_y_list[-2] + accel_y_list[-1]) / 2) * dt
    # ZVU untuk Y
    if zvu_counter_y >= zvu_step_hold:
        v_y_new = 0.0
    velocity_y_list.append(v_y_new)
    y_new = position_y_list[-1] + v_y_new * dt
    position_y_list.append(y_new)

    # Print log
    print(f"Step: {len(accel_x_list)} | "
          f"Accel X: {ax_ms2:.3f} m/s², Vx: {v_x_new:.3f} m/s, X: {x_new:.3f} m, ZVU_x: {zvu_counter_x} | "
          f"Accel Y: {ay_ms2:.3f} m/s², Vy: {v_y_new:.3f} m/s, Y: {y_new:.3f} m, ZVU_y: {zvu_counter_y}")

    # Update plot
    steps = range(len(position_x_list))
    line_vx.set_data(steps, velocity_x_list)
    line_vy.set_data(steps, velocity_y_list)
    line_px.set_data(steps, position_x_list)
    line_py.set_data(steps, position_y_list)
    for ax in axs:
        ax.set_xlim(0, max(100, len(position_x_list)))
    axs[0].set_ylim(
        min(min(velocity_x_list), min(velocity_y_list))-0.2, 
        max(max(velocity_x_list), max(velocity_y_list))+0.2
    )
    axs[1].set_ylim(
        min(min(position_x_list), min(position_y_list))-0.2,
        max(max(position_x_list), max(position_y_list))+0.2
    )
    return line_vx, line_vy, line_px, line_py

ani = FuncAnimation(fig, update, interval=int(dt*1000))

try:
    print("Tekan Ctrl+C untuk berhenti.")
    plt.show(block=True)
except KeyboardInterrupt:
    print("Stop.")
    plt.close()

# Save data jika butuh
np.savez('imu_trapezoidal_integration_xy.npz',
         accel_x=accel_x_list,
         velocity_x=velocity_x_list,
         position_x=position_x_list,
         accel_y=accel_y_list,
         velocity_y=velocity_y_list,
         position_y=position_y_list)
