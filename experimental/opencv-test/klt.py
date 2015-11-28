import numpy as np
import cv2
import os.path
import time
import colorcorrect.algorithm as cca
import pandas as pd

feature_params = dict( 
        maxCorners = 20,
        qualityLevel = 0.1,
        minDistance = 7,
        blockSize = 7,
        )
# Parameters for lucas kanade optical flow
lk_params = dict( 
        winSize  = (15,15),
        maxLevel = 2,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )



m_dir = os.path.dirname(__file__)
face_model_file = os.path.join(m_dir, 'models/haarcascade_frontalface_default.xml')
eye_model_file = os.path.join(m_dir, 'models/haarcascade_eye.xml')
profile_model_file = os.path.join(m_dir, 'models/haarcascade_profileface.xml')
nose_model_file = os.path.join(m_dir, 'models/nariz.xml')


FRAME_RESIZE = (320, 240)
face_cascade = cv2.CascadeClassifier(face_model_file)
eye_cascade = cv2.CascadeClassifier(eye_model_file)
nose_cascade = cv2.CascadeClassifier(nose_model_file)
profile_cascade = cv2.CascadeClassifier(profile_model_file)


def display_and_wait():

    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        return False
    return True

def get_features(gray):

    faces = face_cascade.detectMultiScale(image = gray, 
            scaleFactor = 1.3, 
            minNeighbors = 3, 
            flags = 0, 
            minSize= (50,50))
    eyes = []
    noses = []

    for (x,y,w,h) in faces:

        roi_23down = gray[y+(h/3):y+h, x:x+w]
        roi_23up = gray[y:y+(2*h/3), x:x+w]

        eyes = eye_cascade.detectMultiScale(image = roi_23up, 
            scaleFactor = 1.1, 
            minNeighbors = 5, 
            flags = 0)

        noses = nose_cascade.detectMultiScale(image = roi_23down, 
            scaleFactor = 1.1, 
            minNeighbors = 3, 
            flags = 0)

    return (faces, eyes, noses)

def find_new_KLT(cap):

    face_found = False
    nose_mask = []
    old_gray = []
    p0_nose = []

    while(not face_found):

        ret,frame = cap.read()
        if ret == True:

            frame = cca.stretch(cv2.resize(frame, FRAME_RESIZE))
            cv2.imshow('frame',frame)

            if (not display_and_wait()):
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            (faces, eyes, noses) = get_features(gray)
            nose_mask = np.zeros(gray.shape)

            for (x,y,w,h) in faces:
            
                # Detect the face and save to DF
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0), 2)

                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)

                print "Found face " + str(len(eyes)) + " " + str(len(noses))

                for (nx,ny,nw,nh) in noses:
                    cv2.rectangle(frame,(x+nx,y+ny),(x+nx+nw,y+ny+nh),(0,0,255),2)

                if (len(noses) == 1):

                    face_found = True
                    old_frame = frame

                    """
                    for (ex,ey,ew,eh) in eyes:
                        
                        # ex and ey are relative to the face frame, need to shift to entire frame
                        tx = ex+x
                        ty = ey+y
                        eye_mask[ty:ty+eh, tx:tx+ew] = 1.
                    """

                    for (nx,ny,nw,nh) in noses:

                        # ex and ey are relative to the face frame, need to shift to entire frame
                        tx = nx+x
                        ty = ny+y+h/3
                        nose_mask[ty:ty+nh, tx:tx+nw] = 1.

                    nose_mask = nose_mask.astype(np.uint8)
                    break

    # Take first frame and find corners in it
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    #p0_eyes = cv2.goodFeaturesToTrack(old_gray, mask = eye_mask, **feature_params)
    p0_nose = cv2.goodFeaturesToTrack(old_gray, mask = nose_mask, **feature_params)

    return (old_gray, frame, p0_nose)

def getOneEvent(cap, old_gray, p0_nose):

    ERROR_ALLOWANCE = 5
    ret = False;

    while (ret == False):
        ret, frame = cap.read()

        if (ret == True):

            # Within each iteration, there are 3 outcomes:
            # 1. We have KLT points which are stable and within the Viola-Area
            # 2. We have no KLT points, but we have a face. Initialize KLT.

            frame = cca.stretch(cv2.resize(frame, FRAME_RESIZE))
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

            ### KLT : Optical Flow
            # calculate optical flow

            while (p0_nose == None or len(p0_nose) < 3):               
                (old_gray, frame, p0_nose) = find_new_KLT(cap)

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1_nose, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_nose, None, **lk_params)

            good_new_nose = p1_nose[st==1]
            good_old_nose = p0_nose[st==1]

            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new_nose,good_old_nose)):
                a,b = new.ravel()
                c,d = old.ravel()
                cv2.line(frame, (a,b),(c,d),(0,255,255), 2)
                cv2.circle(frame,(a,b),5,(0,255,255),-1)
                 
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()        
            p0_nose = good_new_nose.reshape(-1,1,2)

            this_event.update(dict(
                noseX=np.mean(p0_nose, axis=0)[0][0],
                noseY=np.mean(p0_nose, axis=0)[0][1]
                ))

            ################################################################################

            ### Viola-Jones : Regional Detection
            (faces, eyes, noses) = get_features(frame_gray)
                    
            if (len(faces) == 1):

                (x,y,w,h) = faces[0]
                    
                this_event.update(dict(
                    isFrontFace=1,
                    faceLeft=x,
                    faceRight=(x+w),
                    faceTop=(y),
                    faceBottom=(y+h)
                    ))

                # Detect the face and save to DF
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0), 2)

                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)

                if (len(noses) == 1):
                    
                    (nx,ny,nw,nh) = noses[0]

                    i = 0
                    p0_nose_filter = np.ones(len(p0_nose), dtype=bool)

                    while i < len(p0_nose):    
                        px = p0_nose[i][0][0]
                        py = p0_nose[i][0][1]
                        
                        if not(px > x+nx-ERROR_ALLOWANCE and px < x+nx+nw+ERROR_ALLOWANCE) \
                            or not(py > y+ny+(h/3)-ERROR_ALLOWANCE and py < y+ny+nh+(h/3)+ERROR_ALLOWANCE):
                            p0_nose_filter[i] = False

                        i += 1

                    p0_nose = p0_nose[p0_nose_filter]
                    cv2.rectangle(frame,(x+nx,y+ny+(h/3)),(x+nx+nw,y+ny+nh+(h/3)),(0,0,255),2)

    cv2.imshow('frame',frame)
    display_and_wait()

    print(this_event)
    return (this_event, old_gray, p0_nose)

def run():

    if (os.path.isfile(face_model_file) == True):
        print 'Face model found!' 
    else:
        print 'Face model NOT found!' 

    if (os.path.isfile(eye_model_file) == True):
        print 'Eye model found!' 
    else:
        print 'Eye model NOT found!' 


    if (os.path.isfile(profile_model_file) == True):
        print 'Profile model found!'     
    else:
        print 'Profile model NOT found!'     

    if (os.path.isfile(nose_model_file) == True):
        print 'Nose model found!'     
    else:
        print 'Nose model NOT found!'     

    cap = cv2.VideoCapture(0)
    events = []

    (old_gray, frame, p0_nose) = find_new_KLT(cap)

    while(1):

        (this_event, old_gray, p0_nose) = getOneEvent(cap, old_gray, p0_nose)
        events.append(this_event)

    df = pd.DataFrame(events)
    cv2.destroyAllWindows()
    cap.release()
    return df

if __name__ == '__main__':
    df = run() 
    df.to_csv('driven.out')
