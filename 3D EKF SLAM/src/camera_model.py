import numpy as np

class CameraModel:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def project(self, points):
        x, y, z = points[:,0], points[:,1], points[:,2]
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        return np.stack([u,v], axis=-1)
