import cv2
import logging
import time
import numpy as np
import colorcorrect.algorithm as cca
import sys
sys.path.append('../experimental/opencv-test')

import klt

from sensor import BaseSensor 
from sensor import SensorMaster


class CameraSensor(BaseSensor):
    
    def __init__(self, dir_path, camera, sensor_name):
        BaseSensor.__init__(self,dir_path, sensor_name)
        # Camera needs to initiated in the main thread
        self.camera = camera
        self.face_model_file = 'models/haarcascade_frontalface_default.xml'
        self.eye_model_file = 'models/haarcascade_eye.xml'
        self.profile_model_file = 'models/haarcascade_profileface.xml'
        self.nose_model_file = 'models/nariz.xml'

        self.frame_index = 0
        self.p0_nose = []
        self.old_frame = []

    def init_sensor(self):
        logging.debug('Initializing Camera')
        self._init_viola()

    def _init_viola(self):
        self.face_cascade = cv2.CascadeClassifier(self.face_model_file)
        self.eye_cascade = cv2.CascadeClassifier(self.eye_model_file)
        self.nose_cascade = cv2.CascadeClassifier(self.nose_model_file)
        self.profile_cascade = cv2.CascadeClassifier(self.profile_model_file)

    def read_sensor(self):
            
        """
        Will try and extract a vector from the frame
        """
        (row, self.frame_index, self.old_frame, self.p0_nose) = \
                klt.getOneEvent(
                        self.camera, 
                        self.frame_index, 
                        self.old_frame, 
                        self.p0_nose,
                        )
        row['timestamp'] = time.time()
        
        return row


    def filter(self, df):
        return df

    def process(self,df):
        return df

    def metric(self, df, init_values):
        pass

if __name__ == '__main__':
    sensors = SensorMaster()

    sensors.add_sensor( 
            CameraSensor('test', 0)
            )

    import pdb; pdb.set_trace()
    sensors.sample_sensors()
