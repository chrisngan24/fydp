import abc
import cv2
import time
import threading
import os
import logging

import pandas as pd
import serial


 
class SensorMaster:
    """
    Thread manager
    Samples each sensor in different threads
    manages when to save the data and how to kill the threads
    """
    def __init__(self):
        self.sensors = [] # each sensor inherits threading.Thread

    def add_sensor(self, sensor_thread):
        """
        Add a new sensor thread to manage

        %sensor_thread% - the BaseSensor object that represents the interface to the sensor
        """
        i = len(self.sensors)
        self.sensors.append(sensor_thread)
        self.sensors[i].daemon = True

    def save_sensors(self):
        """
        Saves data in each sensor thread
        """
        for sensor in self.sensors:
            sensor.save()
            
    def stop_sensors(self):
        """
        Stops the sensor thread
        """
        for sensor in self.sensors:
            sensor.stop()
            

    def sample_sensors(self, callback = lambda sensors: None, **kwargs):
        """
        Start sampling data from the sensors
        callback must take the SEnsorMaster as a parameter
        """
        try:
            for sensor in self.sensors:
                sensor.start()
            while True:
                pass
        except KeyboardInterrupt:
            # hack only use the keyboard interrupt
            self.stop_sensors()
            time.sleep(1)
            self.save_sensors()
            files = map(lambda x: x.file_name, self.sensors)
            callback(files, **kwargs)

class BaseSensor(threading.Thread):
    """
    A base class that all sensor's can inherit froms. 
    Outlines the threading approach
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, dir_path, sensor_name):
        """
        Give the directory path to save to
        %dir_path% - directory to save data to
        %sensor_name% - the str constant that is the sensor
        """
        threading.Thread.__init__(self)
        # raw data points
        self.data_store = []
        self.dir_path = dir_path
        self.sensor_name = sensor_name
        self.file_name = '%s/%s.csv' % (self.dir_path, self.sensor_name)

    def stop(self):
        self.is_running = False


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
        return - a hash map with the data
        """
        pass 

    @abc.abstractmethod
    def filter(self, df):
        """
        After data is collected
        how to filter out the data frame
        """
        pass

    @abc.abstractmethod
    def process(self, df):
        """
        After data is collected, how to process
        the panda data frame
        """
        pass

    @abc.abstractmethod
    def metric(self, df, init_values):
        """
        After processing and filter,
        compute some steady state metric values
        """
        pass

    def run(self):
        """
        Some while loop that keeps on reading from the thread
        """
        print('Starting %s' % self.sensor_name)
        self.init_sensor()
        self.is_running = True
        while(self.is_running):
            data_hash = self.read_sensor()
            self.data_store.append(data_hash)
        print 'Kill sensor', self.sensor_name

    def save(self):
        """
        Save current data to a csv file
        """
        if not os.path.exists(self.dir_path) and not os.path.isdir(self.dir_path):
            os.mkdir(self.dir_path)

        df = pd.DataFrame(self.data_store)
        print df.head()
        df = self.filter(df)
        df = self.process(df)
        df.to_csv(self.file_name, index=False)
