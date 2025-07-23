######################################################
# Copyright (c) 2021 Maker Portal LLC
# Adapted for FaBo9Axis_MPU9250 by ChatGPT
######################################################
#
# This code reads data from the MPU9250 board
# via FaBo9Axis_MPU9250 and solves for
# calibration coefficients for the accelerometer
#
######################################################

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import FaBo9Axis_MPU9250

# --- Setup IMU
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

def accel_cal(cal_size=1000):
    print("-" * 50)
    print("Accelerometer Calibration")
    mpu_offsets = [[], [], []] # offset array to be printed
    axis_vec = ['x', 'y', 'z'] # axis labels
    cal_directions = ["upward", "downward", "perpendicular to gravity"] # direction for IMU cal
    cal_indices = [0, 1, 2] # axis indices (FaBo: x=0, y=1, z=2)
    for qq, ax_qq in enumerate(axis_vec):
        ax_offsets = [[], [], []]
        print("-" * 50)
        for direc_ii, direc in enumerate(cal_directions):
            input("-" * 8 + f" Press Enter and Keep IMU Steady to Calibrate the {ax_qq}-axis pointed {direc}")
            # Clear buffer between readings
            [get_accel() for _ in range(50)]
            mpu_array = []
            while len(mpu_array) < cal_size:
                try:
                    ax, ay, az = get_accel()
                    mpu_array.append([ax, ay, az])
                except:
                    continue
            ax_offsets[direc_ii] = np.array(mpu_array)[:, cal_indices[qq]]
        # Use three calibrations (+1g, -1g, 0g) for linear fit
        popts, _ = curve_fit(
            accel_fit,
            np.append(np.append(ax_offsets[0], ax_offsets[1]), ax_offsets[2]),
            np.append(np.append(1.0 * np.ones(np.shape(ax_offsets[0])),
                               -1.0 * np.ones(np.shape(ax_offsets[1]))),
                      0.0 * np.ones(np.shape(ax_offsets[2]))),
            maxfev=10000)
        mpu_offsets[cal_indices[qq]] = popts # place slope and intercept in offset array
    print('Accelerometer Calibrations Complete')
    return mpu_offsets

if __name__ == '__main__':
    #
    ###################################
    # Accelerometer Gravity Calibration
    ###################################
    #
    accel_labels = ['a_x', 'a_y', 'a_z']
    cal_size = 1000 # number of points to use for calibration (bisa dikurangi jika terlalu lama)
    accel_coeffs = accel_cal(cal_size=cal_size) # grab accel coefficients

    ###################################
    # Record new data 
    ###################################
    data = np.array([get_accel() for ii in range(0, cal_size)]) # new values

    ###################################
    # Plot with and without offsets
    ###################################
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 1, figsize=(12, 9))
    for ii in range(0, 3):
        axs[0].plot(data[:, ii],
                    label='${}$, Uncalibrated'.format(accel_labels[ii]))
        axs[1].plot(accel_fit(data[:, ii], *accel_coeffs[ii]),
                    label='${}$, Calibrated'.format(accel_labels[ii]))
    axs[0].legend(fontsize=14)
    axs[1].legend(fontsize=14)
    axs[0].set_ylabel('$a_{x,y,z}$ [g]', fontsize=18)
    axs[1].set_ylabel('$a_{x,y,z}$ [g]', fontsize=18)
    axs[1].set_xlabel('Sample', fontsize=18)
    axs[0].set_ylim([-2, 2])
    axs[1].set_ylim([-2, 2])
    axs[0].set_title('Accelerometer Calibration Correction', fontsize=18)
    fig.savefig('accel_calibration_output.png', dpi=300,
                bbox_inches='tight', facecolor='#FCFCFC')
    plt.show()
    print("\nHasil koefisien kalibrasi (m, b) untuk x, y, z:")
    for i, coeff in enumerate(accel_coeffs):
        print(f"{accel_labels[i]}: slope={coeff[0]:.6f}, offset={coeff[1]:.6f}")
