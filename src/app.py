from flask import Flask,render_template, request
app = Flask(__name__, static_url_path='')

import json
import pandas as pd
import os
import sys
        

import cv2
import subprocess
import time
import threading

import runner

####
# Threading
###
class RunnerThread(threading.Thread):
    def __init__(
       self, 
       video_port = 0,
       wheel_port ='/dev/cu.usbmodem1421', 
       session_name = int(time.time())
       ):
       threading.Thread.__init__(self)
       self.video_port = video_port
       self.wheel_port = wheel_port
       self.session_name = session_name
       self.is_running =False
       self.cap = None
       #self.sensors = None
       #self._stop = threading.Event()
       self.cap = cv2.VideoCapture(1)

    
    def run(self):
        self.run_thread()
        #raise KeyboardInterrupt
        #self.cap.release()
    
    def run_thread(self):
        self.is_running = True
        if self.cap != None:
            print 'trying to release'
        print 'starting new camera'
        #self.cap = cv2.VideoCapture(int(self.video_port))
        #self.cap = cv2.VideoCapture(1)
        #runner.run_runner(self.video_port, self.wheel_port, self.session_name)
        self.sensors, kwargs = runner.run_runner(self.cap, self.wheel_port, self.session_name)
        self.sensors.sample_sensors(**kwargs)

    def stop_thread(self):
        print 'Throwing keyboard error'
        self.is_running = False
        self.sensors.stop_sensors()
        self.cap.release()
        #self._stop.set()
        
    # def thread_runner(self, 



runner_thread = None
###
# Actual code
###


def render_home():
    return render_template('home.html')

###
# App Routing
###
@app.route("/")
def route_home():
    return render_home()

runner_thread =  dict(thread=None)

@app.route("/runner", methods=['POST'])
def route_runner( ):
    print 'hi'
    if request.method == 'POST':
        with app.app_context():
            #if runner_thread == None:
            app.thread = RunnerThread()
            #runner_thread.video_port = request.form['videoPort']
            #runner_thread.wheel_port = request.form['wheelPort']
            #runner_thread.session_name = request.form['sessionName']
            #runner_thread.cap = cv2.VideoCapture(0)
            app.thread.start()
            return 'ok'

@app.route("/stop", methods=['POST'])
def route_stop( ):
    if request.method == 'POST':
        with app.app_context():
            print 'trying to stop'

            app.thread.stop_thread()
            app.thread.join()
            print '======\n\n'

            
            os.execl(__file__, '')
            return 'ok'
        
    

if __name__ == "__main__":
    app.run(
        
        port=5001,
        )
