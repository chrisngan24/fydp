import cv2
import logging
import copy

from sensors import sensor, wheel_sensor, camera_sensor
from optparse import OptionParser

import pandas as pd
import numpy as np
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# A mock run_fusion callback function for rebuilding
def mock_fusion(
        files, 
        has_camera=True, 
        has_wheel=True,
        data_direc='',
        write_results=True,
        is_move_video=True,
        ):

    print "Finished building:"
    print files

if __name__ == '__main__':
    
    parser = OptionParser()
    parser.add_option('-b', '--BuildName', default=None)
    (options, args) = parser.parse_args()

    if options.BuildName == None:
        print "No BuildName given. Give as a -b argument"
        exit()

    # Use buildname to name the sensor. All the output files will be CAMERA-BuildName.csv
    rebuild_name = "CAMERA-" + options.BuildName
    testing_dir = 'test_suite/test_cases/'
    test_case_list = sorted(next(os.walk(testing_dir))[1])
    print test_case_list

    # Find all tests, and rebuild one at a time
    for test in test_case_list:
        for fi in os.listdir(testing_dir + test):
            if fi.find('drivelog_temp.avi') == 0:

                data_direc = testing_dir + test
                video_name = data_direc + '/' + fi
                cap = cv2.VideoCapture(video_name)

                sensors = sensor.SensorMaster()
                sensors.add_sensor(
                        camera_sensor.CameraSensor(
                            data_direc,
                            cap,
                            rebuild_name,
                            )
                        )

                sensors.sample_sensors(
                            callback=mock_fusion,
                            has_camera=True,
                            has_wheel=False,
                            data_direc=data_direc,
                            )

    print "Rebuilding was completed succesfully."
