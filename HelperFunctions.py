import numpy as np


def quaternRotate(v, q):
    rows, col = v.shape
    z = np.zeros((1, col))
    return quaternion_multiply(quaternion_multiply(q, np.concatenate((z, v), axis=0)), quaternConj(q))[1:, :]


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    h = np.array([- np.multiply(x1, x0) - np.multiply(y1, y0) - np.multiply(z1, z0) + np.multiply(w1, w0),
                  np.multiply(x1, w0) + np.multiply(y1, z0) - np.multiply(z1, y0) + np.multiply(w1, x0),
                  - np.multiply(x1, z0) + np.multiply(y1, w0) + np.multiply(z1, x0) + np.multiply(w1, y0),
                  np.multiply(x1, y0) - np.multiply(y1, x0) + np.multiply(z1, w0) + np.multiply(w1, z0)],
                 dtype=np.float64)
    return h


def quaternConj(quaternion):
    return np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])
