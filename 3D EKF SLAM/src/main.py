import os
import numpy as np
from imu_ekf import predict_state_3d
from landmark_ekf import update_landmarks_3d
from visualize_3d import Visualizer3D

# ---------------- Dataset Setup ----------------
dataset_number = 14
data_dir = '../data/'
data_file = os.path.join(data_dir, f'{dataset_number:02d}.npz')
os.makedirs(data_dir, exist_ok=True)

# Generate fake dataset if missing
if not os.path.exists(data_file):
    N, M = 50, 5  # steps, landmarks
    linear_velocity = np.random.rand(N,3)*0.1
    angular_velocity = np.random.rand(N,3)*0.01
    landmarks = np.random.rand(M,3)*5
    features = np.tile(landmarks, (N,1,1)) + np.random.randn(N,M,3)*0.05
    np.savez(data_file,
             linear_velocity=linear_velocity,
             angular_velocity=angular_velocity,
             features=features)
    print(f"Fake dataset {data_file} created!")

# ---------------- Load Dataset ----------------
data = np.load(data_file)
linear_vel = data['linear_velocity']
angular_vel = data['angular_velocity']
features = data['features']
N, M, _ = features.shape

# ---------------- EKF Initialization ----------------
state_dim = 6 + 3*M
mu = np.zeros(state_dim)
Sigma = np.eye(state_dim) * 0.01
Q = np.eye(6)*0.0001
R = np.eye(3)*0.05
dt = 0.01

# ---------------- Visualizer ----------------
save_folder = '../outputs/images'
os.makedirs(save_folder, exist_ok=True)
viz = Visualizer3D(save_folder, prefix=f'dataset_{dataset_number}')

# ---------------- EKF-SLAM Loop ----------------
for t in range(N):
    # IMU Prediction
    mu, Sigma = predict_state_3d(mu, Sigma, linear_vel[t], angular_vel[t], Q, dt)

    # Simulated LiDAR
    lidar_measurements = features[t,:,:] + np.random.randn(M,3)*0.05

    # Landmark Update
    mu, Sigma = update_landmarks_3d(mu, Sigma, lidar_measurements, R)

    # Update Visualizer (live)
    viz.update(mu, [mu[6+3*i:6+3*i+3] for i in range(M)],
               linear_vel=linear_vel, original_landmarks=features[0,:,:3])

# ---------------- Save Final Figures ----------------
viz.save_final()
print(f"3D EKF-SLAM completed. Images saved in '{save_folder}'.")
