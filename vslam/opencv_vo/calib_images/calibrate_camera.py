# calibrate_camera.py
import cv2
import numpy as np
import glob
import os

CHECKERBOARD = (9, 6)  # Jumlah titik sudut internal
square_size = 1.0      # Misal 1.0 satuan; sesuaikan kalau kamu tahu ukuran sebenarnya

# Persiapkan titik 3D real-world (misalnya z = 0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # Titik 3D dunia nyata
imgpoints = []  # Titik 2D dari citra

# Ambil semua path gambar
images = glob.glob("dataset/*.png")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        print(f"✔️ Found corners in {fname}")
    else:
        print(f"❌ No corners in {fname}")

# Kalibrasi kamera
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Simpan ke file
np.savez("calib_result.npz", K=K, dist=dist)

print("\n🎯 Calibration matrix (K):")
print(K)
print("\n📦 Distortion coefficients:")
print(dist)
