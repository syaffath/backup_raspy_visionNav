# camera_poses.py
import os
import numpy as np
import cv2

class CameraPoses:
    def __init__(self, intrinsic, skip_frames=1, log_path=None):
        self.K = intrinsic
        self.extrinsic = np.array(((1,0,0,0),(0,1,0,0),(0,0,1,0)))
        self.P = self.K @ self.extrinsic
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        self.world_points = []
        self.skip_frames = skip_frames

        # Trajectory & logging
        self.cur_pose = np.concatenate((np.identity(3), np.zeros((3,1))), axis=1) # 3x4 pose
        self.old_frame = None
        self.frame_idx = 0
        self.path_xyz = []
        self.log_path = log_path
        if self.log_path:
            self.logfile = open(self.log_path, "w")
            self.logfile.write("frame,x,y,z\n")
        else:
            self.logfile = None

    def __del__(self):
        if hasattr(self, "logfile") and self.logfile:
            self.logfile.close()

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, img1, img2):
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        if kp1 is None or kp2 is None: return None, None
        if len(kp1) > 6 and len(kp2) > 6:
            matches = self.flann.knnMatch(des1, des2, k=2)
            good_matches = []
            try:
                for m, n in matches:
                    if m.distance < 0.5 * n.distance:
                        good_matches.append(m)
            except ValueError:
                pass
            q1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            q2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            return q1, q2
        return None, None

    def get_pose(self, q1, q2):
        E, mask = cv2.findEssentialMat(q1, q2, self.K)
        R, t = self.decomp_essential_mat_old(E, q1, q2)
        transf = self._form_transf(R, np.squeeze(t))
        return transf

    def decomp_essential_mat_old(self, E, q1, q2):
        def sum_z_cal_relative_scale(R, t):
            T = self._form_transf(R, t)
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = np.matmul(T, hom_Q1)
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
            sum_of_pos_z_Q1 = sum(Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(Q2[2, :] > 0)
            relative_scale = np.mean(
                np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1) /
                np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1)
            )
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale
        self.world_points.append(t)
        return [R1, t]

    def step_with_frame(self, new_frame):
        """
        Proses satu frame baru. Harus dipanggil setiap loop dari main dengan frame kamera.
        Update pose jika old_frame sudah ada.
        Return: pose saat ini (np.array shape (3,4))
        """
        self.frame_idx += 1
        pose_updated = False
        if self.old_frame is not None:
            q1, q2 = self.get_matches(self.old_frame, new_frame)
            # (opsional: filter is_motion_significant(q1, q2) jika ingin stabilitas)
            if q1 is not None and len(q1) > 20 and len(q2) > 20:
                transf = self.get_pose(q1, q2)
                self.cur_pose = self.cur_pose @ transf
                pose_updated = True
        self.old_frame = new_frame
        # Logging
        if pose_updated:
            x, y, z = self.cur_pose[0, 3], self.cur_pose[1, 3], self.cur_pose[2, 3]
            self.path_xyz.append([x, y, z])
            if self.logfile:
                self.logfile.write(f"{self.frame_idx},{x},{y},{z}\n")
        return self.cur_pose

    def is_stuck(self, frame1, frame2, min_kp=3, min_movement=2.0):
        kp1, des1 = self.orb.detectAndCompute(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = self.orb.detectAndCompute(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), None)

        if kp1 is None or kp2 is None or len(kp1) < min_kp or len(kp2) < min_kp:
            return True  # sangat sedikit fitur, bisa jadi stuck di tembok
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) == 0:
            return True
        diffs = [np.linalg.norm(np.array(kp1[m.queryIdx].pt) - np.array(kp2[m.trainIdx].pt)) for m in matches]
        mean_move = np.mean(diffs)
        return mean_move < min_movement

    def get_xyz(self):
        """Get current x,y,z position (translation only)."""
        return self.cur_pose[0, 3], self.cur_pose[1, 3], self.cur_pose[2, 3]

    def get_path(self):
        return np.array(self.path_xyz)
    
    def close(self):
        if hasattr(self, "logfile") and self.logfile:
            self.logfile.close()
