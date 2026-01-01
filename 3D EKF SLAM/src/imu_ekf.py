import numpy as np

def predict_state_3d(mu, Sigma, v, w, Q, dt):
    """
    Predict robot pose in 3D using IMU
    mu: state [x,y,z,roll,pitch,yaw, landmarks...]
    Sigma: covariance
    v: linear velocity
    w: angular velocity
    Q: 6x6 process noise for robot pose
    dt: time step
    """
    mu_pred = mu.copy()
    mu_pred[:3] += v*dt
    mu_pred[3:6] += w*dt

    Sigma_pred = Sigma.copy()
    Sigma_pred[:6,:6] += Q  # only robot pose block
    return mu_pred, Sigma_pred
