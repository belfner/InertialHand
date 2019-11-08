import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

samplePeriod = 1/256

startTime = 6
stopTime = 26

df = pd.read_csv('straightLine_CalInertialAndMag.csv',index_col=False)
print(df.columns)

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
filtCutOff = 0.001
b, a = signal.butter(1,(2*filtCutOff)/(1/samplePeriod),'high')
acc_magFilt = signal.filtfilt(b, a,acc_magFilt)

stationary = acc_magFilt < 0.05

plt.plot(accX[2000:-4000],'r')
plt.plot(accY[2000:-4000],'g')
plt.plot(accZ[2000:-4000],'b')
plt.plot(acc_magFilt[2000:-4000],':k')
plt.plot(stationary[2000:-4000],'k', linewidth=2)
plt.show()

