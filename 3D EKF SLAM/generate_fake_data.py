import numpy as np
import os

# Create data folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Parameters
num_datasets = 5  # number of fake datasets to create
N = 50           # number of timesteps
M = 5            # number of landmarks

for i in range(10, 10 + num_datasets):
    # Fake IMU data
    linear_velocity = np.random.rand(N,3) * 0.1
    angular_velocity = np.random.rand(N,3) * 0.01

    # Fake stereo features
    features = np.random.rand(N,M,4) * 10

    # Save as 10.npz, 11.npz, ...
    filename = f"data/{i:02d}.npz"
    np.savez(filename,
             linear_velocity=linear_velocity,
             angular_velocity=angular_velocity,
             features=features)
    
    print(f"Fake dataset {filename} generated!")

print("All fake datasets generated successfully!")
