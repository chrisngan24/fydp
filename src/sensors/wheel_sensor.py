import serial
import time
import logging

from sensor import BaseSensor 
from sensor import SensorMaster

class WheelSensor(BaseSensor):
    def __init__(self, dir_path, port, baudrate = 9600):
        BaseSensor.__init__(self,dir_path, 'WHEEL')
        self.prev_theta = 0
        self.port = port
        self.baudrate = baudrate
        self.prev_timestamp = time.time()

    def init_sensor(self):
        self.serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            )
        values = []
        self.serial.setDTR(1)
        time.sleep(0.25)
        self.serial.setDTR(0)

    def read_sensor(self):
        va = self.serial.readline()[:-2]
        while len(va) < 20:
            # keep reading if va values is no good
            va = self.serial.readline()[:-2]
        row = {
                k[0]: float(k[1]) \
                    for k in \
                    map(lambda x: x.split(':'), va.split(',')) \
                    if len(k) == 2
                    }
        # calculate theta 
        row['theta'] = self.prev_theta + self.prev_timestamp * row['gz']
        self.prev_theta = row['theta']

        return dict(
                timestamp=time.time(),
                gz=row['gz'],
                theta=row['theta'],
                )

if __name__ == '__main__':
    PORT = '/dev/cu.usbmodem1411'
    sensors = SensorMaster()

    sensors.add_sensor( 
            WheelSensor('test', PORT),
            )

    import pdb; pdb.set_trace()
    sensors.sample_sensors()

