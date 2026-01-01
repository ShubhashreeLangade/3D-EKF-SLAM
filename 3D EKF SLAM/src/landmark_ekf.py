import numpy as np

def update_landmarks_3d(mu, Sigma, z, R):
    """
    Update landmarks using EKF
    mu: full state
    Sigma: full covariance
    z: measured landmarks [[x,y,z],...]
    R: 3x3 measurement noise
    """
    M = z.shape[0]
    for i in range(M):
        idx = 6 + 3*i
        y = z[i] - mu[idx:idx+3]  # innovation
        S = Sigma[idx:idx+3, idx:idx+3] + R
        K = Sigma[idx:idx+3, idx:idx+3] @ np.linalg.inv(S)
        mu[idx:idx+3] += K @ y
        Sigma[idx:idx+3, idx:idx+3] = (np.eye(3) - K) @ Sigma[idx:idx+3, idx:idx+3]
    return mu, Sigma
