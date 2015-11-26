import cv2
import logging
import time
import numpy as np
import colorcorrect.algorithm as cca

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

        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 20,
                               qualityLevel = 0.1,
                               minDistance = 7,
                               blockSize = 7 )

        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.ERROR_ALLOWANCE = 5

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

    def get_features(self, gray):

        faces = self.face_cascade.detectMultiScale(image = gray, 
                scaleFactor = 1.3, 
                minNeighbors = 3, 
                flags = 0, 
                minSize= (100,100))
        eyes = []
        noses = []

        for (x,y,w,h) in faces:

            roi_23down = gray[y+(h/3):y+h, x:x+w]
            roi_23up = gray[y:y+(2*h/3), x:x+w]

            eyes = self.eye_cascade.detectMultiScale(image = roi_23up, 
                scaleFactor = 1.1, 
                minNeighbors = 5, 
                flags = 0)

            noses = self.nose_cascade.detectMultiScale(image = roi_23down, 
                scaleFactor = 1.1, 
                minNeighbors = 3, 
                flags = 0)

        return (faces, eyes, noses)

    def find_new_KLT(self):

        face_found = False

        while(not face_found):

            ret,frame = self.camera.read()
            if ret == True:

                frame = cca.stretch(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                (faces, eyes, noses) = self.get_features(gray)

                eye_mask = np.zeros(gray.shape)
                nose_mask = np.zeros(gray.shape)

                for (x,y,w,h) in faces:

                    print "Found face " + str(len(eyes)) + " " + str(len(noses))

                    if (len(eyes) == 2 and len(noses) == 1):

                        face_found = True
                        self.old_frame = frame

                        for (ex,ey,ew,eh) in eyes:
                            
                            # ex and ey are relative to the face frame, need to shift to entire frame
                            tx = ex+x
                            ty = ey+y
                            eye_mask[ty:ty+eh, tx:tx+ew] = 1.

                        for (nx,ny,nw,nh) in noses:

                            # ex and ey are relative to the face frame, need to shift to entire frame
                            tx = nx+x
                            ty = ny+y+h/3
                            nose_mask[ty:ty+nh, tx:tx+nw] = 1.

                        nose_mask = nose_mask.astype(np.uint8)
                        eye_mask = eye_mask.astype(np.uint8)
                        break

        # Take first frame and find corners in it
        old_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)
        self.p0_nose = cv2.goodFeaturesToTrack(old_gray, mask = nose_mask, **self.feature_params)

    def viola_face(self, frame):
        
        """
        Run viola jones to find the face
        %frame% - the image frame
        """

        # Within each loop, there are 3 outcomes:
        # 1. We have KLT points which are stable and within the Viola-Area
        # 2. We have no KLT points, but we have a face. Initialize KLT.

        frame = cca.stretch(frame)
        this_event = dict(
            time=time.time(),
            isFrontFace=0,
            faceLeft=-1,
            faceRight=-1,
            faceTop=-1,
            faceBottom=-1,
            noseX=-1,
            noseY=-1,
            )
     
        if (len(self.p0_nose) < 3):
            print "Only " + str(len(self.p0_nose)) + " points left. Reinitializing.. " + str(time.time())
            self.find_new_KLT()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
        old_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)        
        p1_nose, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, self.p0_nose, None, **self.lk_params)

        good_new_nose = p1_nose[st==1]
         
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()        
        self.p0_nose = good_new_nose.reshape(-1,1,2)

        this_event.update(dict(
            noseX=np.mean(self.p0_nose, axis=0)[0][0],
            noseY=np.mean(self.p0_nose, axis=0)[0][1]
            ))

        ################################################################################

        ### Viola-Jones : Regional Detection
        (faces,eyes,noses) = self.get_features(frame_gray)

        if (len(faces) == 1):

            (x,y,w,h) = faces[0]

            this_event.update(dict(
                isFrontFace=1,
                faceLeft=x,
                faceRight=(x+w),
                faceTop=(y),
                faceBottom=(y+h)
                ))

            # Define the ROI for eyes and noise
            roi_gray = frame_gray[y:y+h, x:x+w]

            # Detect noses
            noses = self.nose_cascade.detectMultiScale(image = roi_gray, 
                scaleFactor = 1.15, 
                minNeighbors = 8, 
                flags = 0,
                minSize = (20,20))

            if (len(noses) == 1):
            
                (nx,ny,nw,nh) = noses[0]

                i = 0
                p0_nose_filter = np.ones(len(self.p0_nose), dtype=bool)

                while i < len(self.p0_nose):    
                    px = self.p0_nose[i][0][0]
                    py = self.p0_nose[i][0][1]
                    
                    if not(px > x+nx-self.ERROR_ALLOWANCE and px < x+nx+nw+self.ERROR_ALLOWANCE) \
                        or not(py > y+ny-self.ERROR_ALLOWANCE and py < y+ny+nh+self.ERROR_ALLOWANCE):
                        p0_nose_filter[i] = False

                    i += 1

                self.p0_nose = self.p0_nose[p0_nose_filter]

        print(this_event)
        return this_event

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
