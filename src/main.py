import cv2
import logging

from sensors import sensor, wheel_sensor, camera_sensor
from analysis.head_annotator import HeadAnnotator
from analysis.lane_annotator import LaneAnnotator
import fusion
import visualization
import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

GYRO_PORT = '/dev/cu.usbmodem1411'
#GYRO_PORT = '/dev/ttyACM0'
VIDEO_PORT = 1

data_direc = ''
model_direc = 'models'

def visualize(df, events_hash={}):
    gs = gridspec.GridSpec(2, 1)
    gs.update(hspace=0.5, right=0.8)

    plt.style.use('ggplot')

    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0,:])
    ax2 = fig.add_subplot(gs[1,:])

    visualization.make_line_plot(
        ax1,
        df,
       'timestamp_x',
       ['noseX'],
       title='Position of Nose',
       ylabel='Nose X-coord in the Frame',
       xlabel='# of Samples',
       )

    visualization.make_line_plot(
        ax2,
        df,
        'timestamp_x',
        ['theta'],
        title='Angle of Wheel',
        ylabel='Theta (degrees)',
        xlabel='# of Samples',
        )

    visualization.mark_event(
        ax1,
        events_hash['head_turns'],
        )

    visualization.mark_event(
        ax2,
        events_hash['lane_changes'],
        )

    plt.savefig('%s/%s.png' % (data_direc, 'fused_plot'))


def analyze(df):
    analysis.Analysis(model_direc, data_direc).run('dtw')

def run_fusion(sensors):
    """
    Callback function that
    runs fusion on the two data
    csv files
    """
    files = map(lambda x: x.file_name, sensors.sensors)
    print files
    df = fusion.fuse_csv(files)
    df.to_csv('%s/fused.csv' % data_direc)

    head_events_hash =  HeadAnnotator().annotate_events(df)
    lane_events_hash = LaneAnnotator().annotate_events(df)

    visualize(df, { "head_turns": head_events_hash, "lane_changes": lane_events_hash })

if __name__ == '__main__':
    sensors = sensor.SensorMaster()
    now = time.time()
    data_direc = 'data/%s' % int(now)
    # need to initiate openCV2 in the main thread
    camera = cv2.VideoCapture(VIDEO_PORT)
    sensors.add_sensor(
            camera_sensor.CameraSensor(
                data_direc,
                camera,
                )
            )
    sensors.add_sensor(
            wheel_sensor.WheelSensor(
                data_direc,
                GYRO_PORT,
                )
            )
    # sample the sensors, and fuse data as a callback
    sensors.sample_sensors(callback=run_fusion)
