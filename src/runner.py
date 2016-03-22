import cv2
import logging
import shutil

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
        is_interact=True,
        move_to_app=False,
        interactive_video='drivelog_temp.avi',
        ):
    """
    Callback function that
    runs fusion on the two data
    csv files
    """
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

        for i in xrange(len(head_events_list)):
            head_events_list[i] = head_events_list[i] + (head_events_sentiment[i][0],)
            print head_events_list

    if has_wheel:
        ###
        # All events that are dependent on the steering wheel
        ###
        lane_events_hash, lane_events_list = LaneAnnotator(data_direc).annotate_events(df)

    if has_wheel and has_camera:
        slc = SignalLaneClassifier(df, lane_events_list, head_events_list, head_events_hash, head_events_sentiment)
        lane_events_sentiment = slc.classify_signals()

        for i in xrange(len(lane_events_list)):
            lane_events_list[i] = lane_events_list[i] + (lane_events_sentiment[i][0],)

    #### Compute sentiment classifications

    # annotate the video
    print "Creating video report....."
    video_index = 'frameIndex'
    metadata_file = 'annotated_metadata.json'
    #interactive_video = "annotated_fused.avi"    

    # Created a fused video if possible
    if (is_move_video and has_camera and has_wheel):
        print head_events_list
        print lane_events_list
        final_fused_video = annotation.annotate_video(
                'drivelog_temp.avi',
                interactive_video,
                map(lambda (s, e, t, sent): \
                        (df.loc[s, video_index], df.loc[e, video_index], t, sent),
                        head_events_list),
                map(lambda (s, e, t, sent): \
                        (df.loc[s, video_index], df.loc[e, video_index], t, sent),
                        lane_events_list),
                metadata_file
                )

        move_video(final_fused_video, data_direc)
        move_video(metadata_file, data_direc)

    # Otherwise, create the two seperate ones
    else:
        if (is_move_video and has_camera):
            # I MAY HAVE BROKE THIS @chris
            print head_events_list
            
            final_head_video = annotation.annotate_video(
                    'drivelog_temp.avi', 
                    interactive_video, 
                    map(lambda (s, e, t, sent): \
                            (df.loc[s, video_index], df.loc[e, video_index], t, sent),
                            head_events_list),
                    [],
                    metadata_file
                    )

            move_video(final_head_video, data_direc)
            move_video(metadata_file, data_direc)

        elif (is_move_video and has_wheel and len(lane_events_list) > 0): 
            
            print lane_events_list
            final_lane_video = annotation.annotate_video(
                    'drivelog_temp.avi', 
                    interactive_video, 
                    [],
                    map(lambda (s, e, t, sent): \
                            (df.loc[s, video_index], df.loc[e, video_index], t, sent),
                            lane_events_list),
                    metadata_file
                    )

            move_video(final_lane_video, data_direc)
            move_video(metadata_file, data_direc)

        else:
            
            final_plain_video = annotation.annotate_video(
                'drivelog_temp.avi',
                interactive_video,
                [],
                [],
                metadata_file
                )

    # Also copy drivelog_temp
    if (is_move_video and has_camera):
        move_video('drivelog_temp.avi', data_direc)

    video_name = os.path.join(data_direc, interactive_video)
    if (move_to_app):

        # Convert video 
        convert_command = 'ffmpeg -i ' + video_name + ' ' + data_direc + '/annotated_fused.mp4'
        os.system(convert_command)
        time.sleep(1)

        # Replace most recent, and add to data dir
        shutil.rmtree('../app/static/data/recent', ignore_errors = True)
        time.sleep(1)
        shutil.copytree(data_direc, '../app/static/data/recent')
        time.sleep(1)
        dir_name = data_direc.split('/')[-1]
        shutil.copytree(data_direc, '../app/static/data/' + dir_name)

    if (has_camera and has_wheel and write_results):
        print "Plotting...."
        vis = Visualize(
                        df,
                        {
                            "head_turns": head_events_list, 
                            "lane_changes": lane_events_list,
                            "head_sentiment": head_events_sentiment,
                            "lane_sentiment": lane_events_sentiment
                        },
                        video_name=video_name,
                        data_direc=data_direc
            )
        vis.visualize(is_interact=is_interact)

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
    parser.add_option('-n', '--SessionName', default=None)
    (options, args) = parser.parse_args()
    
    ####
    # Set argument parameters
    ####
    print options
    # flags for controlling if will use camera or wheel
    has_camera = options.VideoPort != None
    has_wheel  = options.WheelPort != None
    
    data_direc = ''
    if options.SessionName != None:
        data_direc = 'data/%s' % options.SessionName
    else:
        now = time.time()
        data_direc = 'data/%s' % int(now)

    sensors = sensor.SensorMaster()

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
                is_interact=False,
                move_to_app=True,
                interactive_video='annotated_fused.avi',
                )
    else:
        print 'No sensors... stopping'
