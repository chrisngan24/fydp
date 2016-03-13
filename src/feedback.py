import cv2
import logging
import copy

from analysis.head_annotator import HeadAnnotator
from analysis.lane_annotator import LaneAnnotator
from optparse import OptionParser

import runner 
import annotation

import pandas as pd
import numpy as np
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def main(session_dir, build_name = None):

    # df = pd.read_csv(path_to_test_video + '/fused.csv')
    print build_name
    if (build_name == None):
        files = [ session_dir + '/WHEEL.csv',
                  session_dir + '/CAMERA.csv',
                  ]
    else:
        files = [ session_dir + '/WHEEL.csv',
                  session_dir + '/CAMERA-' + build_name + '.csv',
                  ]    

    analysis_results = runner.run_fusion(
            files, 
            has_camera=True,
            has_wheel=True,
            data_direc=session_dir,
            is_interact=True,
            is_move_video=False,
            interactive_video='drivelog_temp.avi',
            )
   
if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-s', '--SessionDirectory', default=None)
    parser.add_option('-b', '--BuildName', default=None)
    (options, args) = parser.parse_args()

    main(options.SessionDirectory, options.BuildName) 
    
