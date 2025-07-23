######################################################
# Copyright (c) 2021 Maker Portal LLC
# Adapted for FaBo9Axis_MPU9250 by ChatGPT & user
######################################################

import time, sys
sys.path.append('../')
import numpy as np
import csv, datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz
from scipy import signal
import FaBo9Axis_MPU9250

# --- Inisialisasi IMU
mpu9250 = FaBo9Axis_MPU9250.MPU9250()
time.sleep(2) # wait for MPU to load and settle

#####################################
# Accel Calibration (gravity)
#####################################
def accel_fit(x_input, m_x, b):
    return (m_x * x_input) + b # fit equation for accel calibration

def get_accel():
    a = mpu9250.readAccel()
    return a['x'], a['y'], a['z']

def get_gyro():
    g = mpu9250.readGyro()
    return g['x'], g['y'], g['z']

def get_mag():
    m = mpu9250.readMagnet()
    return m['x'], m['y'], m['z']

def accel_cal():
    print("-"*50)
    print("Accelerometer Calibration")
    mpu_offsets = [[], [], []] # offset array to be printed
    axis_vec = ['x', 'y', 'z']
    cal_directions = ["upward", "downward", "perpendicular to gravity"]
    cal_indices = [0, 1, 2]
    for qq, ax_qq in enumerate(axis_vec):
        ax_offsets = [[], [], []]
        print("-"*50)
        for direc_ii, direc in enumerate(cal_directions):
            input("-"*8 + f" Press Enter and Keep IMU Steady to Calibrate the Accelerometer with the {ax_qq}-axis pointed {direc}")
            [get_accel() for ii in range(0, 50)]
            mpu_array = []
            while len(mpu_array) < cal_size:
                try:
                    ax, ay, az = get_accel()
                    mpu_array.append([ax, ay, az])
                except:
                    continue
            ax_offsets[direc_ii] = np.array(mpu_array)[:, cal_indices[qq]]
        popts, _ = curve_fit(accel_fit, np.append(np.append(ax_offsets[0], ax_offsets[1]), ax_offsets[2]),
                   np.append(np.append(1.0 * np.ones(np.shape(ax_offsets[0])),
                    -1.0 * np.ones(np.shape(ax_offsets[1]))),
                        0.0 * np.ones(np.shape(ax_offsets[2]))),
                            maxfev=10000)
        mpu_offsets[cal_indices[qq]] = popts
    print('Accelerometer Calibrations Complete')
    return mpu_offsets

def imu_integrator():
    dt_stop = 5

    # --- Ambil bias ax, ay (IMU dalam kondisi diam di awal!)
    print("Ambil bias ax, ay... Pastikan IMU tidak bergerak!")
    N_bias = 200
    ax_list, ay_list = [], []
    for _ in range(N_bias):
        ax, ay, _ = get_accel()
        ax_list.append(accel_fit(ax, *accel_coeffs[0]) * 9.80665)
        ay_list.append(accel_fit(ay, *accel_coeffs[1]) * 9.80665)
        time.sleep(0.005)  # delay biar tidak terlalu cepat sampling
    bias_ax = np.mean(ax_list)
    bias_ay = np.mean(ay_list)
    print(f"Bias ax = {bias_ax:.5f} m/s², Bias ay = {bias_ay:.5f} m/s² (akan dikompensasi)")

    plt.style.use('ggplot')
    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(12, 9))

    while True:
        # --- Penampung data
        accel_x, accel_y, t_array = [], [], []
        gyro_array = []
        mag_array = []

        [axs[ii].clear() for ii in range(0, 3)]
        t0 = time.time()
        while True:
            try:
                ax, ay, az = get_accel()
                t_now = time.time() - t0
                t_array.append(t_now)
                # --- Kompensasi bias!
                ax_corr = accel_fit(ax, *accel_coeffs[0]) * 9.80665 - bias_ax
                ay_corr = accel_fit(ay, *accel_coeffs[1]) * 9.80665 - bias_ay
                accel_x.append(ax_corr)
                accel_y.append(ay_corr)
            except:
                continue
            if time.time() - t0 > dt_stop:
                break

        # --- Filter (opsional, sama seperti sebelumnya)
        Fs_approx = len(accel_x) / dt_stop if len(accel_x) > 1 else 1
        if len(accel_x) > 10:
            b_filt, a_filt = signal.butter(4, 5, 'low', fs=Fs_approx)
            accel_x = signal.filtfilt(b_filt, a_filt, accel_x)
            accel_y = signal.filtfilt(b_filt, a_filt, accel_y)
        
        # --- Integrasi velocity & displacement
        vel_x = np.append(0.0, cumtrapz(accel_x, x=t_array))
        vel_y = np.append(0.0, cumtrapz(accel_y, x=t_array))
        disp_x = np.append(0.0, cumtrapz(vel_x, x=t_array))
        disp_y = np.append(0.0, cumtrapz(vel_y, x=t_array))

        # --- Plot (3 subplot, 2 warna per subplot)
        axs[0].cla(); axs[1].cla(); axs[2].cla()
        axs[0].plot(t_array, accel_x, 'b-', label="a_x")
        axs[0].plot(t_array, accel_y, 'r-', label="a_y")
        axs[0].set_ylabel('Accel (m/s²)')
        axs[0].legend(); axs[0].grid()
        axs[0].set_ylim(-4, 4)

        axs[1].plot(t_array, vel_x, 'b-', label="v_x")
        axs[1].plot(t_array, vel_y, 'r-', label="v_y")
        axs[1].set_ylabel('Velocity (m/s)')
        axs[1].legend(); axs[1].grid()
        axs[1].set_ylim(-4, 4)

        axs[2].plot(t_array, disp_x, 'b-', label="d_x")
        axs[2].plot(t_array, disp_y, 'r-', label="d_y")
        axs[2].set_ylabel('Displacement (m)')
        axs[2].set_xlabel('Time [s]')
        axs[2].legend(); axs[2].grid()
        axs[2].set_ylim(-5, 5)

        axs[0].set_title("MPU9250 Accelerometer Integration (ax, ay, vx, vy, dx, dy)")
        plt.tight_layout()
        plt.pause(0.01)

        # --- Print log hanya displacement
        print(f"Displacement: x={disp_x[-1]:.3f} m, y={disp_y[-1]:.3f} m")

"""
def imu_integrator_realtime():
    print("Ambil bias ax, ay... Pastikan IMU tidak bergerak!")
    N_bias = 200
    ax_list, ay_list = [], []
    for _ in range(N_bias):
        ax, ay, _ = get_accel()
        ax_list.append(accel_fit(ax, *accel_coeffs[0]) * 9.80665)
        ay_list.append(accel_fit(ay, *accel_coeffs[1]) * 9.80665)
        time.sleep(0.005)
    bias_ax = np.mean(ax_list)
    bias_ay = np.mean(ay_list)
    print(f"Bias ax = {bias_ax:.5f} m/s², Bias ay = {bias_ay:.5f} m/s² (akan dikompensasi)")

    plt.style.use('ggplot')
    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(12, 9))

    # --- FULL DATA, NO BUFFER WINDOW!
    accel_x_full, accel_y_full, t_array_full = [], [], []
    t0 = time.time()

   # ZUPT parameter
    window_size = 9

    while True:
        try:
            ax, ay, az = get_accel()
            t_now = time.time() - t0
            ax_corr = accel_fit(ax, *accel_coeffs[0]) * 9.80665 - bias_ax
            ay_corr = accel_fit(ay, *accel_coeffs[1]) * 9.80665 - bias_ay
            ax_corr = round(ax_corr, 1)
            ay_corr = round(ay_corr, 1)
            accel_x_full.append(ax_corr)
            accel_y_full.append(ay_corr)
            t_array_full.append(t_now)

            # --- Filter Butterworth pada seluruh data
            if len(accel_x_full) > 10:
                Fs_approx = 1 / 0.02
                b_filt, a_filt = signal.butter(4, 5, 'low', fs=Fs_approx)
                accel_x_f = signal.filtfilt(b_filt, a_filt, accel_x_full)
                accel_y_f = signal.filtfilt(b_filt, a_filt, accel_y_full)
            else:
                accel_x_f = np.array(accel_x_full)
                accel_y_f = np.array(accel_y_full)

            # --- Integrasi velocity & displacement (cumulative from t=0)
            if len(accel_x_f) > 2:
                vel_x = np.append(0.0, cumtrapz(accel_x_f, x=t_array_full))
                vel_y = np.append(0.0, cumtrapz(accel_y_f, x=t_array_full))

                # ------- ZUPT (Zero Velocity Update) --------
                if len(accel_x_full) >= window_size:
                    is_stationary = (
                        np.all(np.abs(accel_x_full[-window_size:]) <= 0.1) and
                        np.all(np.abs(accel_y_full[-window_size:]) <= 0.1)
                    )
                    if is_stationary:
                        vel_x[:] = 0
                        vel_y[:] = 0
                # --------------------------------------------

                disp_x = np.append(0.0, cumtrapz(vel_x, x=t_array_full))
                disp_y = np.append(0.0, cumtrapz(vel_y, x=t_array_full))
            else:
                vel_x = np.zeros_like(accel_x_f)
                vel_y = np.zeros_like(accel_y_f)
                disp_x = np.zeros_like(accel_x_f)
                disp_y = np.zeros_like(accel_y_f)

            print(vel_x)

            axs[0].cla(); axs[1].cla(); axs[2].cla()
            axs[0].plot(t_array_full, accel_x_f, 'b-', label="a_x")
            axs[0].plot(t_array_full, accel_y_f, 'r-', label="a_y")
            axs[0].set_ylabel('Accel (m/s²)')
            axs[0].legend(); axs[0].grid(); axs[0].set_ylim(-2, 2)

            axs[1].plot(t_array_full, vel_x, 'b-', label="v_x")
            axs[1].plot(t_array_full, vel_y, 'r-', label="v_y")
            axs[1].set_ylabel('Velocity (m/s)')
            axs[1].legend(); axs[1].grid(); axs[1].set_ylim(-2, 2)

            axs[2].plot(t_array_full, disp_x, 'b-', label="d_x")
            axs[2].plot(t_array_full, disp_y, 'r-', label="d_y")
            axs[2].set_ylabel('Displacement (m)')
            axs[2].set_xlabel('Time [s]')
            axs[2].legend(); axs[2].grid(); axs[2].set_ylim(-5, 5)

            axs[0].set_title("MPU9250 Accelerometer Integration (ax, ay, vx, vy, dx, dy)")
            plt.tight_layout()
            plt.pause(0.01)
            time.sleep(0.02)  # 20ms delay

        except KeyboardInterrupt:
            print("\nLive plot dihentikan oleh user.")
            break
        except Exception as e:
            print("Error:", e)
            continue            
"""                
"""
def kinematic():
    print("Ambil bias ax, ay... Pastikan IMU tidak bergerak!")
    N_bias = 200
    ax_list, ay_list = [], []
    dt = 0.02  # asumsikan 20 ms, atau bisa pakai rata-rata selisih waktu antar frame
    for _ in range(N_bias):
        ax, ay, _ = get_accel()
        ax_list.append(accel_fit(ax, *accel_coeffs[0]) * 9.80665)
        ay_list.append(accel_fit(ay, *accel_coeffs[1]) * 9.80665)
        time.sleep(0.005)
    bias_ax = np.mean(ax_list)
    bias_ay = np.mean(ay_list)
    print(f"Bias ax = {bias_ax:.5f} m/s², Bias ay = {bias_ay:.5f} m/s² (akan dikompensasi)")

    plt.style.use('ggplot')
    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(12, 9))

    x_pos = 0.0
    y_pos = 0.0
    v_x = 0.0
    v_y = 0.0

    t_arr = []
    x_arr = []
    y_arr = []
    vx_arr = []
    vy_arr = []
    ax_arr = []
    ay_arr = []

    t0 = time.time()
    while True:
        try:
            t_now = time.time() - t0
            ax, ay, az = get_accel()
            a_x = accel_fit(ax, *accel_coeffs[0]) * 9.80665 - bias_ax
            a_y = accel_fit(ay, *accel_coeffs[1]) * 9.80665 - bias_ay

            a_x = round(a_x, 1)
            a_y = round(a_y, 1)


            # Kinematic equation
            x_pos = x_pos + v_x * dt + 0.5 * a_x * dt * dt
            y_pos = y_pos + v_y * dt + 0.5 * a_y * dt * dt
            v_x = v_x + a_x * dt
            v_y = v_y + a_y * dt

            t_arr.append(t_now)
            x_arr.append(x_pos)
            y_arr.append(y_pos)
            vx_arr.append(v_x)
            vy_arr.append(v_y)
            ax_arr.append(a_x)
            ay_arr.append(a_y)

            # Live plot update
            axs[0].cla()
            axs[0].plot(t_arr, ax_arr, 'b-', label="a_x")
            axs[0].plot(t_arr, ay_arr, 'r-', label="a_y")
            axs[0].set_ylabel('Accel (m/s²)')
            axs[0].legend(); axs[0].grid(); axs[0].set_ylim(-5, 5)

            axs[1].cla()
            axs[1].plot(t_arr, vx_arr, 'b-', label="v_x")
            axs[1].plot(t_arr, vy_arr, 'r-', label="v_y")
            axs[1].set_ylabel('Velocity (m/s)')
            axs[1].legend(); axs[1].grid(); axs[1].set_ylim(-1, 1)

            axs[2].cla()
            axs[2].plot(t_arr, x_arr, 'b-', label="x")
            axs[2].plot(t_arr, y_arr, 'r-', label="y")
            axs[2].set_ylabel('Displacement (m)')
            axs[2].set_xlabel('Time [s]')
            axs[2].legend(); axs[2].grid(); axs[2].set_ylim(-2, 2)

            axs[0].set_title("MPU9250 Kinematic Update (Euler)")
            plt.tight_layout()
            plt.pause(0.01)
            time.sleep(dt)

            print(f"x = {x_pos:.3f} m, y = {y_pos:.3f} m, vx = {v_x:.3f} m/s, vy = {v_y:.3f} m/s", end="\r")
        except KeyboardInterrupt:
            print("\nLive plot dihentikan oleh user.")
            break
        except Exception as e:
            print("Error:", e)
            continue
"""

from scipy import signal

def kinematic():
    print("Ambil bias ax, ay... Pastikan IMU tidak bergerak!")
    N_bias = 200
    ax_list, ay_list = [], []
    dt = 0.02  # asumsikan 20 ms sampling
    for _ in range(N_bias):
        ax, ay, _ = get_accel()
        ax_list.append(accel_fit(ax, *accel_coeffs[0]) * 9.80665)
        ay_list.append(accel_fit(ay, *accel_coeffs[1]) * 9.80665)
        time.sleep(0.005)
    bias_ax = np.mean(ax_list)
    bias_ay = np.mean(ay_list)
    print(f"Bias ax = {bias_ax:.5f} m/s², Bias ay = {bias_ay:.5f} m/s² (akan dikompensasi)")

    plt.style.use('ggplot')
    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(12, 9))

    x_pos = 0.0
    y_pos = 0.0
    v_x = 0.0
    v_y = 0.0

    t_arr = []
    x_arr = []
    y_arr = []
    vx_arr = []
    vy_arr = []
    ax_arr = []
    ay_arr = []

    # Untuk filter Butterworth
    window_size = 40    # sesuaikan, makin besar makin halus/makin lambat respons
    fc = 7              # frekuensi cutoff dalam Hz (misal 2-5 Hz untuk IMU)
    b_filt, a_filt = signal.butter(4, fc, 'low', fs=1/dt)
    ax_window = []
    ay_window = []

    t0 = time.time()
    while True:
        try:
            t_now = time.time() - t0
            ax, ay, az = get_accel()
            a_x_raw = accel_fit(ax, *accel_coeffs[0]) * 9.80665 - bias_ax
            a_y_raw = accel_fit(ay, *accel_coeffs[1]) * 9.80665 - bias_ay

            # Masukkan ke window
            ax_window.append(a_x_raw)
            ay_window.append(a_y_raw)
            
            if len(ax_window) > window_size:
                ax_window.pop(0)
                ay_window.pop(0)

            # Filter Butterworth
            padlen = 3 * max(len(a_filt), len(b_filt))  # biasanya 15 utk Butterworth orde-4
            if len(ax_window) > padlen:
                a_x = signal.filtfilt(b_filt, a_filt, ax_window)[-1]
                a_y = signal.filtfilt(b_filt, a_filt, ay_window)[-1]
            else:
                a_x = a_x_raw
                a_y = a_y_raw
            
            a_x = round(a_x, 1)
            a_y = round(a_y, 1)

            # Kinematic equation (Euler integration)
            x_pos = x_pos + v_x * dt + 0.5 * a_x * dt * dt
            y_pos = y_pos + v_y * dt + 0.5 * a_y * dt * dt
            v_x = v_x + a_x * dt
            v_y = v_y + a_y * dt

            t_arr.append(t_now)
            x_arr.append(x_pos)
            y_arr.append(y_pos)
            vx_arr.append(v_x)
            vy_arr.append(v_y)
            ax_arr.append(a_x)
            ay_arr.append(a_y)

            # Live plot update
            axs[0].cla()
            axs[0].plot(t_arr, ax_arr, 'b-', label="a_x")
            axs[0].plot(t_arr, ay_arr, 'r-', label="a_y")
            axs[0].set_ylabel('Accel (m/s²)')
            axs[0].legend(); axs[0].grid(); axs[0].set_ylim(-5, 5)

            axs[1].cla()
            axs[1].plot(t_arr, vx_arr, 'b-', label="v_x")
            axs[1].plot(t_arr, vy_arr, 'r-', label="v_y")
            axs[1].set_ylabel('Velocity (m/s)')
            axs[1].legend(); axs[1].grid(); axs[1].set_ylim(-5, 5)

            axs[2].cla()
            axs[2].plot(t_arr, x_arr, 'b-', label="x")
            axs[2].plot(t_arr, y_arr, 'r-', label="y")
            axs[2].set_ylabel('Displacement (m)')
            axs[2].set_xlabel('Time [s]')
            axs[2].legend(); axs[2].grid(); axs[2].set_ylim(-5, 5)

            axs[0].set_title("MPU9250 Kinematic Update (Euler + Butterworth Filter)")
            plt.tight_layout()
            plt.pause(0.01)
            time.sleep(dt)

            print(f"x = {x_pos:.3f} m, y = {y_pos:.3f} m, vx = {v_x:.3f} m/s, vy = {v_y:.3f} m/s", end="\r")
        except KeyboardInterrupt:
            print("\nLive plot dihentikan oleh user.")
            break
        except Exception as e:
            print("Error:", e)
            continue


if __name__ == '__main__':
    t0 = time.time()
    start_bool = True
    mpu_labels = ['a_x', 'a_y', 'a_z']
    cal_size = 1000
    old_vals_bool = True
    if not old_vals_bool:
        accel_coeffs = accel_cal()
    else:
        accel_coeffs = [np.array([ 0.999545 , -0.018243]),
                        np.array([ 0.997714, -0.017341]),
                        np.array([0.981974 , -0.007192])]
    data = np.array([get_accel() for ii in range(0, cal_size)])
    imu_integrator()
    #imu_integrator_realtime()
    #kinematic()

# Hasil koefisien kalibrasi (m, b) untuk x, y, z:
# a_x: slope=0.999545, offset=-0.018243
# a_y: slope=0.997714, offset=-0.017341
# a_z: slope=0.981974, offset=-0.007192
