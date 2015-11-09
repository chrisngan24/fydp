import cv2

from sensors import sensor, wheel_sensor, camera_sensor
import time

GYRO_PORT = '/dev/cu.usbmodem1411'
VIDEO_PORT = 0

if __name__ == '__main__':
    sensors = sensor.SensorMaster()
    now = time.time()
    data_direc = 'data/%s' % int(now)
    sensors.add_sensor(
            wheel_sensor.WheelSensor(
                data_direc,
                GYRO_PORT,
                )
            )
    # need to initiate openCV2 in the main thread
    camera = cv2.VideoCapture(VIDEO_PORT)
    sensors.add_sensor(
            camera_sensor.CameraSensor(
                data_direc,
                camera,
                )
            )

    sensors.sample_sensors()
