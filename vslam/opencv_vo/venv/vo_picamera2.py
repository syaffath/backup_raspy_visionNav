import cv2
import numpy as np
import csv
import time
import matplotlib.pyplot as plt
from picamera2 import Picamera2

class VOStreamer:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

        # Init camera
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (self.width, self.height)}))
        self.picam2.start()

        # Load calibration
        data = np.load("../calib_images/calib_result.npz")
        self.K = data["K"]
        self.dist = data["dist"]

        # Optical flow state
        self.prev_gray = None
        self.prev_pts = None

        # Pose
        self.position = np.zeros((3, 1))

        # Live plot
        self.x_history = []
        self.z_history = []
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.plot_line, = self.ax.plot([], [], 'bo-', markersize=2)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Z")
        self.ax.set_title("Hist. Visual Odometry")
        self.ax.grid(True)
        self.ax.axis("equal")

    def update_plot(self, x, z):
        self.x_history.append(x)
        self.z_history.append(z)
        self.plot_line.set_data(self.x_history, self.z_history)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self):
        with open("trajectory_log_OF.csv", "w", newline="") as logfile:
            csv_writer = csv.writer(logfile)
            csv_writer.writerow(["frame_id", "x", "y", "z", "t_norm", "avg_flow", "status"])

            frame_id = 0

            while True:
                frame = self.picam2.capture_array()
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                gray = cv2.undistort(gray, self.K, self.dist)
                vis_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) #frame.copy()

                if self.prev_gray is None:
                    self.prev_gray = gray
                    self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=7)
                    continue

                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None)
                if next_pts is None or status is None:
                    continue

                good_prev = self.prev_pts[status.flatten() == 1]
                good_next = next_pts[status.flatten() == 1]

                if len(good_prev) >= 8:
                    for (p1, p2) in zip(good_prev, good_next):
                        x1, y1 = p1.ravel()
                        x2, y2 = p2.ravel()
                        cv2.arrowedLine(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1, tipLength=0.3)

                    flow = good_next - good_prev
                    flow_magnitude = np.linalg.norm(flow, axis=1)
                    avg_flow = np.mean(flow_magnitude)

                    if avg_flow < 0.5:
                        x, y, z = self.position.flatten()
                        print(f"[{frame_id:05}] 🛌 Idle. Pose: x={x:.2f}, z={z:.2f}")
                        csv_writer.writerow([frame_id, x, y, z, 0.0, avg_flow, "idle"])
                    else:
                        E, _ = cv2.findEssentialMat(good_next, good_prev, self.K, method=cv2.RANSAC, threshold=1.0)
                        if E is not None:
                            _, R, t, _ = cv2.recoverPose(E, good_next, good_prev, self.K)
                            t_norm = np.linalg.norm(t)

                            if 0.01 < t_norm < 1.0:
                                self.position += t
                                x, y, z = self.position.flatten()

                                print(f"[{frame_id:05}] ✅ Moved. Pose: x={x:.2f}, z={z:.2f}")
                                csv_writer.writerow([frame_id, x, y, z, t_norm, avg_flow, "updated"])
                                self.update_plot(x, z)
                            else:
                                print(f"[{frame_id:05}] ⚠️ Skipped update (t norm = {t_norm:.4f})")

                self.prev_gray = gray
                self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=7)
                frame_id += 1
                time.sleep(0.03)

                cv2.imshow("Camera View + Optical Flow", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    vo = VOStreamer()
    vo.run()
