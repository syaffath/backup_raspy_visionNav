import cv2
import numpy as np

# Parameter kamera
WIDTH = 640
HEIGHT = 480
FRAME_SIZE = WIDTH * HEIGHT * 3 // 2  # YUV420

# Load kalibrasi
calib = np.load("../calib_images/calib_result.npz")
K = calib["K"]
dist = calib["dist"]

# Inisialisasi pose
R_total = np.eye(3)
t_total = np.zeros((3, 1))

# Buka named pipe
with open("live.yuv", 'rb') as pipe:
    # Ambil frame pertama
    raw = pipe.read(FRAME_SIZE)
    if len(raw) < FRAME_SIZE:
        print("❌ Failed to read first frame")
        exit()

    # Konversi ke grayscale
    y = np.frombuffer(raw, dtype=np.uint8, count=WIDTH * HEIGHT)
    gray_prev = y.reshape((HEIGHT, WIDTH))
    gray_prev = cv2.undistort(gray_prev, K, dist)
    pts_prev = cv2.goodFeaturesToTrack(gray_prev, maxCorners=200, qualityLevel=0.01, minDistance=7)

    frame_id = 0
    while True:
        raw = pipe.read(FRAME_SIZE)
        if len(raw) < FRAME_SIZE:
            continue

        y = np.frombuffer(raw, dtype=np.uint8, count=WIDTH * HEIGHT)
        gray = y.reshape((HEIGHT, WIDTH))
        gray = cv2.undistort(gray, K, dist)

        pts_next, status, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray, pts_prev, None)

        if pts_next is not None and status is not None:
            good_prev = pts_prev[status.flatten() == 1]
            good_next = pts_next[status.flatten() == 1]

            if len(good_prev) >= 8:
                E, _ = cv2.findEssentialMat(good_next, good_prev, K, method=cv2.RANSAC, threshold=1.0)
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, good_next, good_prev, K)

                    t_total += R_total @ t
                    R_total = R @ R_total

                    # Cetak posisi saat ini
                    x, y, z = t_total[0][0], t_total[1][0], t_total[2][0]
                    print(f"[{frame_id:04}] ✅ Pose: x={x:.3f}, y={y:.3f}, z={z:.3f}")

        # Update previous
        gray_prev = gray.copy()
        pts_prev = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=7)
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
