import abc
import cv2
import time
import threading
import os
import logging

import pandas as pd
import serial


 
class SensorMaster:
    def __init__(self):
        self.sensors = [] # each sensor inherits threading.Thread

    def add_sensor(self, sensor_thread):
        i = len(self.sensors)
        self.sensors.append(sensor_thread)
        self.sensors[i].daemon = True

    def save_sensors(self):
        for sensor in self.sensors:
            sensor.save()
        
    def sample_sensors(self):
        try:
            for sensor in self.sensors:
                sensor.start()
            while True:
                pass
        except KeyboardInterrupt:
            # hack only use the keyboard interrupt
            self.save_sensors()
        except e:
            sensor.stop()
            print e

class BaseSensor(threading.Thread):
    __metaclass__ = abc.ABCMeta

    def __init__(self, dir_path, sensor_name):
        threading.Thread.__init__(self)
        self.data_store = []
        self.dir_path = dir_path
        self.sensor_name = sensor_name

    @abc.abstractmethod
    def init_sensor(self):
        """
        How does the sensor connect with this thread
        """
        pass

    @abc.abstractmethod
    def read_sensor(self):
        """
        Read one data value from the sensor
        """
        pass 

    def run(self):
        """
        Some while loop that keeps on reading from the thread
        """
        print('Starting %s' % self.sensor_name)
        self.init_sensor()
        while(True):
            data_hash = self.read_sensor()
            self.data_store.append(data_hash)

    def save(self):
        """
        Save current data to a csv file
        """
        if not os.path.exists(self.dir_path) and not os.path.isdir(self.dir_path):
            os.mkdir(self.dir_path)
        pd.DataFrame(self.data_store).to_csv('%s/%s.csv' %
                (self.dir_path, self.sensor_name)
                )
