import pandas as pd
from scipy import signal
from AHRS import AHRS
import HelperFunctions
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.spatial.transform import Rotation as R
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

samplePeriod = 1/256

startTime = int(6/ samplePeriod)
stopTime = int(26 / samplePeriod)

dfo = pd.read_csv('straightLine_CalInertialAndMag.csv', index_col=False)
# # print(dfo.columns)
df = dfo.iloc[startTime:stopTime + 1]
time = np.array([(x + startTime) / 256 for x in range(len(df))])
gyrX = df['Gyroscope X (deg/s)'].values
gyrY = df['Gyroscope Y (deg/s)'].values
gyrZ = df['Gyroscope Z (deg/s)'].values
accX = df['Accelerometer X (g)'].values
accY = df['Accelerometer Y (g)'].values
accZ = df['Accelerometer Z (g)'].values
# time = df['Seconds'].values
# gyrX = df['GyroX'].values
# gyrY = df['GyroY'].values
# gyrZ = df['GyroZ'].values
# accX = df['AccX'].values
# accY = df['AccY'].values
# accZ = df['AccZ'].values



acc_mag = np.sqrt(np.multiply(accX, accX) + np.multiply(accY, accY) + np.multiply(accZ, accZ))

# High Pass filter accelerometer data
filtCutOff = 0.001
b, a = signal.butter(1, (2 * filtCutOff) / (1 / samplePeriod), 'high')
acc_magFilt = signal.filtfilt(b, a, acc_mag)

# compute absolute value
acc_magFilt = abs(acc_magFilt)

# Low Pass filter accelerometer data
filtCutOff = 5
b, a = signal.butter(1, (2 * filtCutOff) / (1 / samplePeriod), 'low')
acc_magFilt = signal.filtfilt(b, a, acc_magFilt)

stationary = acc_magFilt < 0.10

plt.plot(time,accX,'r',linewidth=.5)
plt.plot(time,accY,'g',linewidth=.5)
plt.plot(time,accZ,'b',linewidth=.5)
plt.plot(time,acc_magFilt,':k',linewidth=.5)
# plt.plot(time,stationary,'k', linewidth=2)
plt.show()

# -------------------------------------------------------------------------
# Compute orientation
quat = []

AHRSalgorithm = AHRS()

# Initial convergence
initPeriod = 2
gyroHold = np.array([0, 0, 0])
index = int(initPeriod / samplePeriod) + 1
index = 8
accHold = [np.mean(accX[:index]), np.mean(accY[:index]), np.mean(accZ[:index])]
for x in range(2000):
    AHRSalgorithm.UpdateIMU(gyroHold, accHold)

for i in range(len(accX)):
    if stationary[i]:
        AHRSalgorithm.Kp = 0.5
    else:
        AHRSalgorithm.Kp = 0

    quat.append(AHRSalgorithm.UpdateIMU(np.radians([gyrX[i], gyrY[i], gyrZ[i]]), [accX[i], accY[i], accZ[i]]))

# -------------------------------------------------------------------------
# Compute translational accelerations

# Rotate body accelerations to Earth frame

print(AHRSalgorithm.Quaternion)
quat = np.array(quat)
acc = HelperFunctions.quaternRotate(np.array([accX, accY, accZ]), HelperFunctions.quaternConj(quat.T))

# Convert acceleration measurements to m/s/s
acc = acc * 9.81

plt.figure("Acceleration")
plt.title('Accelerations')
plt.plot(time, acc[0], 'r', linewidth=.5)
plt.plot(time, acc[1], 'g', linewidth=.5)
plt.plot(time, acc[2], 'b', linewidth=.5)
plt.xlim([time[0], time[-1]])
plt.xticks(np.arange(time[0], time[-1] + .45, 5))
plt.legend(['X', 'Y', 'Z'])
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s/s)')
plt.show()

# -------------------------------------------------------------------------
# Compute translational velocities

acc[2] = acc[2] - 9.98

vel = np.zeros(acc.shape)
for i in range(1, vel.shape[1]):
    vel[:, i] = vel[:, i - 1] + acc[:, i] * samplePeriod
    if stationary[i]:
        vel[:, i] = [0, 0, 0]

# Compute integral drift during non-stationary periods
velDrift = np.zeros(vel.shape)
d = np.diff(stationary.astype(int))
stationaryStart = np.where(d == -1)[0]
stationaryEnd = np.where(d == 1)[0]

for x in range(len(stationaryEnd)):
    driftRate = vel[:, stationaryEnd[x] - 1] / (stationaryEnd[x] - stationaryStart[x])
    enum = np.array(list(range(1, stationaryEnd[x] - stationaryStart[x] + 1)))
    drift = np.array([enum * driftRate[0], enum * driftRate[1], enum * driftRate[2]])
    velDrift[:,stationaryStart[x]+1:stationaryEnd[x]+1] = drift
t = 0

# Remove integral drift
vel = vel - velDrift

plt.figure("Velocity")
plt.title('Velocity ')
plt.plot(time, vel[0], 'r', linewidth=.5)
plt.plot(time, vel[1], 'g', linewidth=.5)
plt.plot(time, vel[2], 'b', linewidth=.5)
plt.xlim([time[0], time[-1]])
plt.xticks(np.arange(time[0], time[-1] + .45, 5))
plt.legend(['X', 'Y', 'Z'])
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.show()

# -------------------------------------------------------------------------
# Compute translational position

# Integrate velocity to yield position

pos = np.zeros(vel.shape)

for i in range(1, pos.shape[1]):
    pos[:, i] = pos[:, i - 1] + vel[:, i] * samplePeriod

plt.figure("Position")
plt.title('Position')
plt.plot(time, pos[0], 'r', linewidth=.5)
plt.plot(time, pos[1], 'g', linewidth=.5)
plt.plot(time, pos[2], 'b', linewidth=.5)
plt.xlim([time[0], time[-1]])
plt.xticks(np.arange(time[0], time[-1] + .45, 5))
plt.legend(['X', 'Y', 'Z'])
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')

# ax = fig.gca(projection='3d')
# ax.set_aspect('equal')
# ax.plot(pos[0],pos[1],pos[2])
# plt.show()

# plt.figure()
# fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
# plt.ion()
#
# def generate_data(num):
#     return R.from_quat(quat[num]).as_dcm()
#
# r = generate_data(0)
#
# quiverx = ax.quiver(0, 0, 0, r[0][0], r[0][1], r[0][2], pivot="tail", color="red")
# quivery = ax.quiver(0, 0, 0, r[1][0], r[1][1], r[1][2], pivot="tail", color="green")
# quiverz = ax.quiver(0, 0, 0, r[2][0], r[2][1], r[2][2], pivot="tail", color="blue")
#
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(-1, 1)
#
#
#
# for i in range(2000,quat.shape[0],10):
#     # global quiverx
#     # global quivery
#     # global quiverz
#     quiverx.remove()
#     quivery.remove()
#     quiverz.remove()
#     r = generate_data(i)
#     quiverx = ax.quiver(0, 0, 0, r[0][0], r[0][1], r[0][2], pivot="tail", color="red")
#     quivery = ax.quiver(0, 0, 0, r[1][0], r[1][1], r[1][2], pivot="tail", color="green")
#     quiverz = ax.quiver(0, 0, 0, r[2][0], r[2][1], r[2][2], pivot="tail", color="blue")
#     plt.draw()
#     plt.pause(0.1)
#
# plt.show(block=True)

# r = generate_data(0)
#
#
# quiverx = ax.quiver(0, 0, 0, r[0][0], r[0][1], r[0][2], pivot="tail", color="red")
# quivery = ax.quiver(0, 0, 0, r[1][0], r[1][1], r[1][2], pivot="tail", color="green")
# quiverz = ax.quiver(0, 0, 0, r[2][0], r[2][1], r[2][2], pivot="tail", color="blue")
#
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(-1, 1)
#
# def update(num):
#     global quiverx
#     global quivery
#     global quiverz
#     quiverx.remove()
#     quivery.remove()
#     quiverz.remove()
#     r = generate_data(num)
#     quiverx = ax.quiver(0, 0, 0, r[0][0], r[0][1], r[0][2], pivot="tail", color="red")
#     quivery = ax.quiver(0, 0, 0, r[1][0], r[1][1], r[1][2], pivot="tail", color="green")
#     quiverz = ax.quiver(0, 0, 0, r[2][0], r[2][1], r[2][2], pivot="tail", color="blue")
#
#
# ani = animation.FuncAnimation(fig, update, frames=range(2000,quat.shape[0]), interval=100)
# plt.show()

# fig = plt.figure()
# ax = p3.Axes3D(fig)
#
# def gen(n,data):
#     i = 0
#     while i<n:
#         yield data[i]
#         i+=1
#
# def update(num, data, line):
#     line.set_data(data[:2, :num])
#     line.set_3d_properties(data[2, :num])
#
# data = np.array(list(gen(1000,pos.T[900:]))).T
#
# line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])
#
# # Setting the axes properties
# ax.set_xlim3d([-1.0, 20.0])
# ax.set_xlabel('X')
#
# ax.set_ylim3d([-1.0, 20.0])
# ax.set_ylabel('Y')
#
# ax.set_zlim3d([-1.0, 20.0])
# ax.set_zlabel('Z')
#
# ani = animation.FuncAnimation(fig, update, 1000, fargs=(data, line), interval=1, blit=False)
# #ani.save('matplot003.gif', writer='imagemagick')
# plt.show()