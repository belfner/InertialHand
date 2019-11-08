import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from AHRS import AHRS

samplePeriod = 1/256

startTime = int(6/samplePeriod)
stopTime = int(26/samplePeriod)

dfo = pd.read_csv('straightLine_CalInertialAndMag.csv',index_col=False)
print(dfo.columns)
df = dfo.iloc[startTime:stopTime+1]
time = np.array([x/256 for x in range(len(df))])[startTime:stopTime+1]
gyrX = df['Gyroscope X (deg/s)'].values
gyrY = df['Gyroscope Y (deg/s)'].values
gyrZ = df['Gyroscope Z (deg/s)'].values
accX = df['Accelerometer X (g)'].values
accY = df['Accelerometer Y (g)'].values
accZ = df['Accelerometer Z (g)'].values


acc_mag = np.sqrt(np.multiply(accX,accX)+np.multiply(accY,accY)+np.multiply(accZ,accZ))

# High Pass filter accelerometer data
filtCutOff = 0.001
b, a = signal.butter(1,(2*filtCutOff)/(1/samplePeriod),'high')
acc_magFilt = signal.filtfilt(b, a,acc_mag)

#compute absolute value
acc_magFilt = abs(acc_magFilt)

# Low Pass filter accelerometer data
filtCutOff = 5
b, a = signal.butter(1,(2*filtCutOff)/(1/samplePeriod),'low')
acc_magFilt = signal.filtfilt(b, a,acc_magFilt)

stationary = acc_magFilt < 0.05

# plt.plot(time,accX,'r')
# plt.plot(time,accY,'g')
# plt.plot(time,accZ,'b')
# plt.plot(time,acc_magFilt,':k')
# plt.plot(time,stationary,'k', linewidth=2)
# plt.show()

#-------------------------------------------------------------------------
#Compute orientation
quat = np.zeros((len(time), 4))

AHRSalgorithm = AHRS()

#Initial convergence
initPeriod = 2
gyroHold = np.array([0, 0, 0])
index = int(initPeriod/samplePeriod)+1
accHold = [np.mean(accX[:index]), np.mean(accY[:index]), np.mean(accZ[:index])]
for x in range(2000):
    AHRSalgorithm.UpdateIMU(gyroHold, accHold)
print(AHRSalgorithm.Quaternion)


