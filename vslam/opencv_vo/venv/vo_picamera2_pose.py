import cv2
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from picamera2 import Picamera2


class LiveVisualOdometry:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

        # Init Picamera2
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (self.width, self.height)}))
        self.picam2.start()
        time.sleep(1.0)

        # Calibration
        data = np.load("../calib_images/calib_result.npz")
        self.K = data["K"]
        self.dist = data["dist"]

        # ORB + FLANN
        self.orb = cv2.ORB_create(3000)
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

        self.prev_kp = None
        self.prev_des = None
        self.prev_gray = None

        self.cur_pose = np.eye(4)
        self.x_history = []
        self.z_history = []

        # Live plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.plot_line, = self.ax.plot([], [], 'bo-', markersize=2)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Z")
        self.ax.set_title("Live VO with ORB")
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

    def get_pose(self, q1, q2):
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)
        if E is None:
            return None
        _, R, t, _ = cv2.recoverPose(E, q1, q2, self.K)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T

    def run(self):
        with open("trajectory_log_orb.csv", "w", newline="") as logfile:
            csv_writer = csv.writer(logfile)
            csv_writer.writerow(["frame_id", "x", "y", "z"])
            frame_id = 0

            while True:
                frame = self.picam2.capture_array()
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                gray = cv2.undistort(gray, self.K, self.dist)
                vis = frame.copy()

                kp, des = self.orb.detectAndCompute(gray, None)
                if self.prev_gray is not None and des is not None and self.prev_des is not None:
                    matches = self.flann.knnMatch(self.prev_des, des, k=2)
                    good = []
                    for m_n in matches:
                        if len(m_n) == 2:
                            m, n = m_n
                            if m.distance < 0.8 * n.distance:
                                good.append(m)


                    if len(good) > 8:
                        q1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good])
                        q2 = np.float32([kp[m.trainIdx].pt for m in good])
                        pose = self.get_pose(q1, q2)
                        if pose is not None:
                            self.cur_pose = self.cur_pose @ np.linalg.inv(pose)
                            x, z = self.cur_pose[0, 3], self.cur_pose[2, 3]
                            self.update_plot(x, z)
                            print(f"[{frame_id:05}] ✅ x={x:.2f}, z={z:.2f}")
                            csv_writer.writerow([frame_id, x, 0.0, z])

                        for m in good:
                            pt1 = tuple(np.round(self.prev_kp[m.queryIdx].pt).astype(int))
                            pt2 = tuple(np.round(kp[m.trainIdx].pt).astype(int))
                            cv2.arrowedLine(vis, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)

                self.prev_kp = kp
                self.prev_des = des
                self.prev_gray = gray

                frame_id += 1
                cv2.imshow("Live ORB VO", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    vo = LiveVisualOdometry()
    vo.run()
