import matplotlib.pyplot as plt
import numpy as np
import os

class Visualizer3D:
    """
    Live 3D EKF-SLAM visualizer with 4 tabs
    - Trajectory + IMU
    - Trajectory Only
    - SLAM vs Landmarks
    - 2D Top View
    """

    def __init__(self, save_folder, prefix='frame'):
        self.save_folder = save_folder
        self.prefix = prefix
        os.makedirs(save_folder, exist_ok=True)

        self.mu_history = []
        self.landmarks_history = []
        self.frame_count = 0
        plt.ion()

        self.fig_traj_imu, self.ax_traj_imu = plt.subplots(subplot_kw={'projection':'3d'})
        self.fig_traj, self.ax_traj = plt.subplots(subplot_kw={'projection':'3d'})
        self.fig_slam, self.ax_slam = plt.subplots(subplot_kw={'projection':'3d'})
        self.fig_2d, self.ax_2d = plt.subplots(figsize=(8,6))

    def update(self, mu, landmarks, linear_vel=None, original_landmarks=None):
        self.mu_history.append(mu.copy())
        self.landmarks_history.append(landmarks)
        traj = np.array(self.mu_history)

        # 1️⃣ Trajectory + IMU
        self.ax_traj_imu.cla()
        self.ax_traj_imu.plot(traj[:,0], traj[:,1], traj[:,2], 'r-', label='Trajectory')
        if linear_vel is not None:
            for t in range(0,len(traj),3):
                self.ax_traj_imu.quiver(traj[t,0], traj[t,1], traj[t,2],
                                        linear_vel[t,0], linear_vel[t,1], linear_vel[t,2],
                                        color='b', length=0.5, normalize=True)
        self.ax_traj_imu.scatter(traj[0,0], traj[0,1], traj[0,2], c='g', s=60, label='Start')
        self.ax_traj_imu.scatter(traj[-1,0], traj[-1,1], traj[-1,2], c='k', s=60, label='End')
        self.ax_traj_imu.set_title('Trajectory + IMU'); self.ax_traj_imu.set_xlabel('X'); self.ax_traj_imu.set_ylabel('Y'); self.ax_traj_imu.set_zlabel('Z'); self.ax_traj_imu.legend()

        # 2️⃣ Trajectory Only
        self.ax_traj.cla()
        self.ax_traj.plot(traj[:,0], traj[:,1], traj[:,2], 'b-', label='Trajectory')
        self.ax_traj.scatter(traj[0,0], traj[0,1], traj[0,2], c='g', s=60, label='Start')
        self.ax_traj.scatter(traj[-1,0], traj[-1,1], traj[-1,2], c='r', s=60, label='End')
        self.ax_traj.set_title('Trajectory Only'); self.ax_traj.set_xlabel('X'); self.ax_traj.set_ylabel('Y'); self.ax_traj.set_zlabel('Z'); self.ax_traj.legend()

        # 3️⃣ SLAM vs Landmarks
        self.ax_slam.cla()
        self.ax_slam.plot(traj[:,0], traj[:,1], traj[:,2], 'r-', label='Trajectory')
        if original_landmarks is not None:
            self.ax_slam.scatter(original_landmarks[:,0], original_landmarks[:,1], original_landmarks[:,2],
                                 c='g', s=50, label='Original Landmarks')
        for lm_set in self.landmarks_history:
            if len(lm_set) > 0:
                lm_arr = np.array(lm_set)
                self.ax_slam.scatter(lm_arr[:,0], lm_arr[:,1], lm_arr[:,2], c='r', marker='x', s=50, label='SLAM Landmarks')
        self.ax_slam.set_title('SLAM vs Landmarks'); self.ax_slam.set_xlabel('X'); self.ax_slam.set_ylabel('Y'); self.ax_slam.set_zlabel('Z'); self.ax_slam.legend()

        # 4️⃣ 2D Top View
        self.ax_2d.cla()
        self.ax_2d.plot(traj[:,0], traj[:,1], 'r-', linewidth=2, label='EKF Trajectory')
        if len(landmarks)>0:
            lm_arr = np.array(landmarks)
            self.ax_2d.scatter(lm_arr[:,0], lm_arr[:,1], c='g', s=30, label='Landmarks')
        self.ax_2d.scatter(traj[0,0], traj[0,1], c='b', s=80, marker='s', label='Start')
        self.ax_2d.scatter(traj[-1,0], traj[-1,1], c='orange', s=80, marker='o', label='End')
        self.ax_2d.set_xlabel('X'); self.ax_2d.set_ylabel('Y'); self.ax_2d.set_title('2D Top View'); self.ax_2d.grid(True); self.ax_2d.legend()

        plt.pause(0.01)

        # Save frames
        for fig, name in zip([self.fig_traj_imu,self.fig_traj,self.fig_slam,self.fig_2d],
                             ['traj_imu','traj','slam','topview']):
            path = os.path.join(self.save_folder, f"{self.prefix}_{name}_{self.frame_count:03d}.png")
            fig.savefig(path)

        self.frame_count += 1

    def save_final(self):
        for fig, name in zip([self.fig_traj_imu,self.fig_traj,self.fig_slam,self.fig_2d],
                             ['traj_imu','traj','slam','topview']):
            path = os.path.join(self.save_folder, f"{self.prefix}_{name}_final.png")
            fig.savefig(path)
        plt.ioff()
        plt.show()
