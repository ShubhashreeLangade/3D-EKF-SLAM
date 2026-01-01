import numpy as np

def generate_fake_measurements(num_steps=50, num_landmarks=5):
    measurements = []
    landmarks_gt = np.random.randn(num_landmarks,3)*5
    for _ in range(num_steps):
        u = np.random.randn(6)*0.1
        z = landmarks_gt + np.random.randn(num_landmarks,3)*0.05
        measurements.append((u, z))
    return measurements, landmarks_gt
