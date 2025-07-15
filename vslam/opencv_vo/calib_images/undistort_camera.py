# undistort_camera.py (versi safe headless)
import cv2
import numpy as np
import glob
import os

# Load parameter kalibrasi
data = np.load("calib_result.npz")
K = data["K"]
dist = data["dist"]

# Buat folder output jika belum ada
os.makedirs("undistorted", exist_ok=True)

# Ambil semua gambar dari dataset
images = glob.glob("dataset/*.png")

for i, fname in enumerate(images):
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, K, dist, None, new_K)

    basename = os.path.basename(fname)
    save_path = os.path.join("undistorted", f"undistorted_{basename}")
    cv2.imwrite(save_path, undistorted)
    print(f"✅ Saved undistorted image to {save_path}")
