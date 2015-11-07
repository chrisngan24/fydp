import numpy as np
import cv2
import os.path
import time
import pandas as pd

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

cap = cv2.VideoCapture(0)

roi_hist = 0;

# setup initial location of window  
# simply hardcoded the values
c,r,w,h = 0,0,150,150
track_window = (c,r,w,h)
events = []

while(1):

    ret,frame = cap.read()
    if ret == True:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.4, 1, 3, (100,100))
        profiles = profile_cascade.detectMultiScale(gray, 1.4, 3, 0)

        this_event = {}

        # Add basic event data
        this_event = dict(
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
            this_event.update(dict(
                faceX=(x+w/2),
                faceY=(y+h/2)
                ))

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            #hsv_roi =  cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
            #roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 180], [0, 180, 0, 180])
            #cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
            #track_window = (x,y,w,h)
            
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.4, 8, 0)

            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            noses = nose_cascade.detectMultiScale(roi_gray, 1.4, 5, 0, (20,20))

            for (nx,ny,nw,nh) in noses:
                cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,0,255),2)
                this_event.update(dict(
                    noseX=(nx+nw/2),
                    noseY=(ny+nh/2)
                    ))

        events.append(this_event)
        cv2.imshow('img',frame)

        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
        elif k == ord('s'):
            cv2.imwrite("viola_snap.jpg",frame)
        elif k == ord('h'):
            plt.show()

    else:
        break

pd.DataFrame(events).to_csv("viola.out")
cv2.destroyAllWindows()
cap.release()