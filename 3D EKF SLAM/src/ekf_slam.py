import numpy as np
from imu_ekf import predict_imu
from landmark_ekf import update_landmarks

class EKF_SLAM_3D:
    def __init__(self):
        # Robot state: x, y, z, roll, pitch, yaw
        self.mu = np.zeros(6)
        self.Sigma = np.eye(6) * 0.01
        self.landmarks_est = []

    def predict(self, u):
        self.mu, self.Sigma = predict_imu(self.mu, self.Sigma, u)

    def update(self, z):
        self.mu, self.Sigma, self.landmarks_est = update_landmarks(self.mu, self.Sigma, z)
