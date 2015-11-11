import cv2
import logging

from sensors import sensor, wheel_sensor, camera_sensor
import fusion
import visualize
import time
import sys
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

#GYRO_PORT = '/dev/cu.usbmodem1411'
GYRO_PORT = '/dev/ttyACM0'
VIDEO_PORT = 0

data_direc = ''

def run_fusion(sensors):
    """
    Callback function that
    runs fusion on the two data
    csv files
    """
    file_1 = sensors.sensors[0].file_name
    df = pd.read_csv(file_1)
    df['timestamp_x'] = df['timestamp']
    '''
    file_2 = sensors.sensors[1].file_name
    df = fusion.fuse_csv(file_1, file_2)
    df.to_csv('%s/fused.csv' % data_direc)
    visualize.make_line_plot(
            df,
            'timestamp_x',
            ['theta', 'gz'],
            file_dir=data_direc,
            title='Angle of Wheel',
            ylabel='Theta (degrees)',
            xlabel='Timestamp (s)',
            )
    '''
    visualize.make_line_plot(
            df,
            'timestamp_x',
            ['noseX'],
            file_dir=data_direc,
            title='',
            ylabel='Nose X-coord in the Frame',
            xlabel='Timestamp (s)',
            )




if __name__ == '__main__':
    sensors = sensor.SensorMaster()
    now = time.time()
    #data_direc = 'data/%s' % 'latest'
    data_direc = 'data/%s' % int(now)
    '''
    sensors.add_sensor(
            wheel_sensor.WheelSensor(
                data_direc,
                GYRO_PORT,
                )
            )
    '''
    # need to initiate openCV2 in the main thread
    camera = cv2.VideoCapture(VIDEO_PORT)
    sensors.add_sensor(
            camera_sensor.CameraSensor(
                data_direc,
                camera,
                )
            )
    # sample the sensors, and fuse data as a callback
    sensors.sample_sensors(callback=run_fusion)
