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
        self.face_model_file = 'models/haarcascade_frontalface_default.xml'
        self.eye_model_file = 'models/haarcascade_eye.xml'
        self.profile_model_file = 'models/haarcascade_profileface.xml'
        self.nose_model_file = 'models/nariz.xml'

    def init_sensor(self):
        logging.debug('Initializing Camera')
        self.init_viola()


    def init_viola(self):
        self.face_cascade = cv2.CascadeClassifier(self.face_model_file)
        self.eye_cascade = cv2.CascadeClassifier(self.eye_model_file)
        self.nose_cascade = cv2.CascadeClassifier(self.nose_model_file)
        self.profile_cascade = cv2.CascadeClassifier(self.profile_model_file)

    def viola_face(self, frame):
        """
        Run viola jones to find the face
        %frame% - the image frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.4, 1, 3, (100,100))
        profiles = self.profile_cascade.detectMultiScale(gray, 1.4, 3, 0)

        row = {}

        # Add basic event data
        row = dict(
            time=time.time(),
            isFrontFace=len(faces),
            isRotatedFace=len(profiles),
            faceX=-1,
            faceY=-1,
            noseX=-1,
            noseY=-1
            )

        for (px,py,pw,ph) in profiles:
            cv2.rectangle(frame,(px,py),(px+pw,py+ph),(255,255,0),2)

        for (x,y,w,h) in faces:
            
            # Detect the face and save to DF
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0), 2)
            row.update(dict(
                faceX=(x+w/2),
                faceY=(y+h/2)
                ))

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.4, 8, 0)

            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            noses = self.nose_cascade.detectMultiScale(roi_gray, 1.4, 5, 0, (20,20))

            for (nx,ny,nw,nh) in noses:
                cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,0,255),2)
                row.update(dict(
                    noseX=(nx+nw/2),
                    noseY=(ny+nh/2)
                    ))
        return row


    def read_sensor(self):
        """
        Will try and extract a vector from the frame
        """
        ret = False
        frame = None
        while ret == False:
            ret,frame = self.camera.read()
        row = self.viola_face(frame)
        row['timestamp'] = time.time()
        return row

if __name__ == '__main__':
    sensors = SensorMaster()

    sensors.add_sensor( 
            CameraSensor('test', 0)
            )

    import pdb; pdb.set_trace()
    sensors.sample_sensors()
