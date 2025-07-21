from mpu9250_jmdev.registers import *
from mpu9250_jmdev.mpu_9250 import MPU9250
import time

# Inisialisasi sensor
mpu = MPU9250(
    address_ak=AK8963_ADDRESS,  # 0x0C
    address_mpu_master=MPU9050_ADDRESS_68,  # 0x68
    address_mpu_slave=None,
    bus=1,  # I2C bus di Raspberry Pi
    gfs=GFS_250,  # Gyro full scale
    afs=AFS_2G,   # Accel full scale
    mfs=AK8963_BIT_16,  # Magnetometer resolution
    mode=AK8963_MODE_C100HZ)

mpu.configure()

while True:
    accel = mpu.readAccelerometerMaster()
    gyro = mpu.readGyroscopeMaster()
    mag = mpu.readMagnetometerMaster()

    print(f"Accel: {accel}, Gyro: {gyro}, Mag: {mag}")
    time.sleep(0.1)
