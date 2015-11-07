import numpy as np
import cv2
import os.path
import time
import pandas as pd

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 20,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

face_model_file = 'models/haarcascade_frontalface_default.xml'
eye_model_file = 'models/haarcascade_eye.xml'
profile_model_file = 'models/haarcascade_profileface.xml'
nose_model_file = 'models/nariz.xml'

if (os.path.isfile(face_model_file) == True):
    print 'Face model found!' 

if (os.path.isfile(eye_model_file) == True):
    print 'Eye model found!' 

if (os.path.isfile(profile_model_file) == True):
    print 'Profile model found!'     

if (os.path.isfile(nose_model_file) == True):
    print 'Nose model found!'     

face_cascade = cv2.CascadeClassifier(face_model_file)
eye_cascade = cv2.CascadeClassifier(eye_model_file)
nose_cascade = cv2.CascadeClassifier(nose_model_file)
profile_cascade = cv2.CascadeClassifier(profile_model_file)

# Create some random colors
color = np.random.randint(0,255,(100,3))

cap = cv2.VideoCapture(0)
face_found = False
p0 = ()
old_frame = []
face_frame = []
face_mask = []


while(not face_found):

    ret,frame = cap.read()
    if ret == True:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 3, 0, (100,100))
        face_mask = np.zeros(gray.shape)

        for (x,y,w,h) in faces:
            
            face_roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_roi, 1.3, 8, 0)
            nose = nose_cascade.detectMultiScale(face_roi, 1.3, 5, 0)

            if (len(eyes) == 2 and len(nose) == 1):

                face_found = True
                old_frame = frame

                for (ex,ey,ew,eh) in eyes:
                    
                    # ex and ey are relative to the face frame, need to shift to entire frame
                    tx = ex+x
                    ty = ey+y
                    face_mask[ty:ty+eh, tx:tx+ew] = 1.

                for (nx,ny,nw,nh) in nose:

                    # ex and ey are relative to the face frame, need to shift to entire frame
                    tx = nx+x
                    ty = ny+y
                    face_mask[ty:ty+nh, tx:tx+nw] = 1.

                face_mask = face_mask.astype(np.uint8)
                break

# Take first frame and find corners in it
#ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = face_mask, **feature_params)

while(1):
    ret,frame = cap.read()

    if ret == True:

        this_event = dict(
            time=time.time(),
            deltaX=-1,
            deltaY=-1
            )

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
     
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            cv2.line(frame, (a,b),(c,d), color[i].tolist(), 2)
            cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
     
        cv2.imshow('frame',frame)
        
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
        elif k == ord('s'):
            cv2.imwrite("klt_snap.jpg",frame)
     
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
