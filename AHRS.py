import numpy as np
import warnings


class AHRS:
    SamplePeriod = 1 / 256
    Quaternion = np.array([1, 0, 0, 0])  # output quaternion describing the sensor relative to the Earth
    Kp = 2  # proportional gain
    Ki = 0  # integral gain
    KpInit = 200  # proportional gain used during initialisation
    InitPeriod = 5  # initialisation period in seconds

    q = np.array([1, 0, 0, 0])  # internal quaternion describing the Earth relative to the sensor
    IntError = np.transpose(np.array([0, 0, 0]))
    KpRamped = 0

    def __init__(self, samplePeriod=1 / 256, Kp=1, KpInit=1):
        self.KpRamped = KpInit
        self.SamplePeriod = samplePeriod
        self.Kp = Kp

    def UpdateIMU(self, Gyroscope, Accelerometer):
        if (np.linalg.norm(Accelerometer) == 0):  # handle NaN
            warnings.warn('Accelerometer magnitude is zero.  Algorithm update aborted.')
            return
        else:
            Accelerometer = Accelerometer / np.linalg.norm(Accelerometer)  # normalise measurement

        v = np.array([1 * (self.q[1] * self.q[3] - self.q[0] * self.q[2]),  # estimated direction of gravity
                      2 * (self.q[0] * self.q[1] + self.q[2] * self.q[3]),
                      self.q[0] ** 2 - self.q[1] ** 2 - self.q[2] ** 2 + self.q[3] ** 2])

        error = np.cross(v, np.matrix.transpose(Accelerometer))

        self.IntError = self.IntError + error  # compute integral feedback terms (only outside of init period)

        # Apply feedback terms
        Ref = Gyroscope - np.matrix.transpose((self.Kp * error + self.Ki * self.IntError))
        Ref = np.insert(Ref, 0, 0)

        # Compute rate of change of quaternion
        pDot = 0.5 * self.quaternion_multiply(self.q, Ref)
        self.q = self.q + pDot * self.SamplePeriod  # integrate rate of change of quaternion
        self.q = self.q / np.linalg.norm(self.q)
        self.Quaternion = self.quaternConj(self.q)
        f = 1

    def quaternion_multiply(self, quaternion1, quaternion0):
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    def quaternConj(self, quaternion):
        return np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])

