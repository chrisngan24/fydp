import cv2
import logging

from sensors import sensor, wheel_sensor, camera_sensor
from analysis.head_annotator import HeadAnnotator
from analysis.lane_annotator import LaneAnnotator
from analysis.signal_head_classifier import SignalHeadClassifier 
from analysis.signal_lane_classifier import SignalLaneClassifier 
import fusion
from visualization import Visualize
import annotation
import time
import sys
import pandas as pd

import os
from optparse import OptionParser


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

GYRO_PORT = '/dev/cu.usbmodem1411'

# For Linux
#GYRO_PORT = '/dev/ttyACM0'
VIDEO_PORT = 1

model_direc = 'models'


def move_video(video_name, data_direc):
    os.rename(video_name, '%s/%s' % (data_direc, video_name))


def run_fusion(
        files, 
        has_camera=True, 
        has_wheel=True,
        data_direc='',
        write_results=True,
        is_move_video=True,
        ):
    """
    Callback function that
    runs fusion on the two data
    csv files
    """
    print has_camera, has_wheel
    print files
    df = fusion.fuse_csv(files)
    if not 'timestamp_x' in df.columns.values.tolist():
        df['timestamp_x'] = df['timestamp']
    if write_results:
        df.to_csv('%s/fused.csv' % data_direc)
    if has_camera:
        ### 
        # All events that are dependent on the camera
        ### 
        head_ann = HeadAnnotator()
        head_events_hash, head_events_list =  head_ann.annotate_events(df)
        shc = SignalHeadClassifier(head_ann.df, head_ann.events)
        head_events_sentiment = shc.classify_signals()

    if has_wheel:
        ###
        # All events that are dependent on the steering wheel
        ###
        lane_events_hash, lane_events_list = LaneAnnotator().annotate_events(df)

    if has_wheel and has_camera:
        slc = SignalLaneClassifier(df, lane_events_list, head_events_list, head_events_hash, head_events_sentiment)
        lane_events_sentiment = slc.classify_signals()



    #### Compute sentiment classifications

    # annotate the video
    print "Creating video report....."
    video_index = 'frameIndex'
    if (is_move_video and has_camera and len(head_events_list) > 0):
        # I MAY HAVE BROKE THIS @chris
        print head_events_list
        final_head_video = annotation.annotate_video(
                'drivelog_temp.avi', 
                'annotated_head.avi', 
                map(lambda (s, e, t): \
                        (df.loc[s, video_index], df.loc[e, video_index], t),
                        head_events_list),
                {'left_turn': (0,255,0), 'right_turn': (255,0,0)},
                )
        move_video(final_head_video, data_direc)
    if (is_move_video and has_wheel and len(lane_events_list) > 0): 
        print lane_events_list
        final_lane_video = annotation.annotate_video(
                'drivelog_temp.avi', 
                'annotated_lane.avi', 
                map(lambda (s, e, t): \
                        (df.loc[s, video_index], df.loc[e, video_index], t),
                        lane_events_list),

                {'left_lane_change': (0,255,0), 'right_lane_change': (255,0,0)},
                )
        move_video(final_lane_video, data_direc)

    if (has_camera and has_wheel and write_results):
        video_name = os.path.join(data_direc, 'annotated_lane.avi')
        print "Plotting...."
        vis = Visualize(
                        df,
                        {
                            "head_turns": head_events_hash, 
                            "lane_changes": lane_events_hash 
                        },
                        video_name=video_name,
                        data_direc=data_direc
            )
        vis.visualize()

    if (has_wheel and has_camera):
        return dict(
                head_events_hash=head_events_hash,
                head_events_list=head_events_list,
                lane_events_hash=lane_events_hash,
                lane_events_list=lane_events_list,
                head_events_sentiment=head_events_sentiment,
                lane_events_sentiment=lane_events_sentiment,
                df=df,
                )
    else:
        return None
    
if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-v', '--VideoPort', default=None)
    parser.add_option('-w', '--WheelPort', default=None)
    (options, args) = parser.parse_args()
    
    ####
    # Set argument parameters
    ####
    print options
    # flags for controlling if will use camera or wheel
    has_camera = options.VideoPort != None
    has_wheel  = options.WheelPort != None
    sensors = sensor.SensorMaster()
    now = time.time()
    data_direc = 'data/%s' % int(now)


    if has_camera:
        video_port = int(options.VideoPort)
        # need to initiate openCV2 in the main thread
        camera = cv2.VideoCapture(video_port)
        sensors.add_sensor(
                camera_sensor.CameraSensor(
                    data_direc,
                    camera,
                    'CAMERA'
                    )
                )
    if has_wheel:
        wheel_port = str(options.WheelPort)
        sensors.add_sensor(
            wheel_sensor.WheelSensor(
                data_direc,
                wheel_port,
                )
            )

    # sample the sensors, and fuse data as a callback
    if len(sensors.sensors) > 0:
        sensors.sample_sensors(
                callback=run_fusion,
                has_camera=has_camera,
                has_wheel=has_wheel,
                data_direc=data_direc,
                )
    else:
        print 'No sensors... stopping'
