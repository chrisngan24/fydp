import numpy as np
import cv2
import os.path
import time
import pandas as pd
import skimage.color

def apply_viola(frame, gray):

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

    # Not used 
    profile_cascade = cv2.CascadeClassifier(profile_model_file)



    # setup initial location of window  
    # simply hardcoded the values
    c,r,w,h = 0,0,150,150
    track_window = (c,r,w,h)



    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = skimage.color.rgb2gray(frame)
    print gray.shape

    faces = face_cascade.detectMultiScale(image = gray, 
            scaleFactor = 1.3, 
            minNeighbors = 5, 
            flags = 0, 
            minSize=(100,100),
            )

    this_event = {}

    # Add basic event data
    this_event = dict(
            time=time.time(),
            isFrontFace=len(faces),
            isRotatedFace=0,
            faceX=-1,
            faceY=-1,
            noseX=-1,
            noseY=-1
            )
    print faces

    for (x,y,w,h) in faces:

        # Detect the face and save to DF
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0), 2)
        this_event.update(dict(
            faceX=(x+w/2),
            faceY=(y+h/2)
            ))

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        roi_23down = gray[y+(h/3):y+h, x:x+w]
        roi_23up = gray[y:y+(2*h/3), x:x+w]

        eyes = eye_cascade.detectMultiScale(image = roi_23up, 
                scaleFactor = 1.1, 
                minNeighbors = 8, 
                flags = 0)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        noses = nose_cascade.detectMultiScale(image = roi_23down, 
            scaleFactor = 1.15, 
            minNeighbors = 8, 
            flags = 0, 
            minSize=(20,20))

        for (nx,ny,nw,nh) in noses:
            cv2.rectangle(roi_color,(nx,ny+(h/3)),(nx+nw,ny+nh+(h/3)),(0,0,255),2)
            this_event.update(dict(
                noseX=(nx+nw/2),
                noseY=(ny+nh/2)
                ))


    return this_event

