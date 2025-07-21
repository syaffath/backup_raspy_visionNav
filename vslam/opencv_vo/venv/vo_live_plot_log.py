# Tambahan untuk live plot dan CSV logging
import matplotlib.pyplot as plt
import csv
import time
import cv2
import numpy as np

# Faktor skala global untuk translasi pose
SCALE_FACTOR = 0.1/2 # Ubah ini ke 1.0 jika tidak ingin diskalakan, ubah dari cm ke meter
MOTION_THRESHOLD = 5.0  # dalam piksel

def initialize_live_plot_and_logging():
    # Logging to CSV
    csv_file = open("vo_log.csv", mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "fps", "x", "y", "z"] + [f"R{i}{j}" for i in range(3) for j in range(3)])

    # Live plot setup
    fig, ax = plt.subplots()
    plt.ion()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Live Trajectory (X-Z)")
    x_data, z_data = [], []

    return fig, ax, x_data, z_data, csv_file, csv_writer


def update_live_plot_and_log(ax, x_data, z_data, pose, fps, csv_writer):
    scaled_x = pose[0, 3] * SCALE_FACTOR
    scaled_y = pose[1, 3] * SCALE_FACTOR
    scaled_z = pose[2, 3] * SCALE_FACTOR

    x_data.append(scaled_x)
    z_data.append(scaled_z)

    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Live Trajectory (X-Z)")
    ax.plot(x_data, z_data, marker='o', linestyle='-')
    plt.draw()
    plt.pause(0.001)

    now = time.time()
    R_flat = pose[:3, :3].flatten().tolist()
    csv_writer.writerow([now, round(fps, 2), scaled_x, scaled_y, scaled_z] + R_flat)


def draw_orb_features(image, keypoints):
    """
    Menampilkan fitur ORB langsung di atas frame kamera.
    """
    return cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)


def detect_orb_features(image):
    orb = cv2.ORB_create(1000)
    keypoints = orb.detect(image, None)
    return keypoints

def is_motion_significant(q1, q2, pixel_thresh=MOTION_THRESHOLD):
    if q1 is None or q2 is None or len(q1) == 0 or len(q2) == 0:
        return False
    displacement = np.linalg.norm(q1 - q2, axis=1)
    mean_disp = np.mean(displacement)
    return mean_disp > pixel_thresh