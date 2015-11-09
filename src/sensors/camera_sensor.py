import cv2
import logging
import time

from sensor import BaseSensor 
from sensor import SensorMaster


class CameraSensor(BaseSensor):
    def __init__(self, dir_path, camera):
        BaseSensor.__init__(self,dir_path, 'CAMERA')
        # Camera needs to initiated in the main thread
        self.camera = camera

    def init_sensor(self):
        logging.debug('Initializing Camera')

    def read_sensor(self):
        ret,frame = self.camera.read()
        # INSERT CODE CFUNG
        return dict(
                timestamp=time.time(),
                )

if __name__ == '__main__':
    sensors = SensorMaster()

    sensors.add_sensor( 
            CameraSensor('test', 0)
            )

    import pdb; pdb.set_trace()
    sensors.sample_sensors()
