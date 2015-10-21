import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# setup initial location of window  
# simply hardcoded the values
c,r,w,h = 0,0,150,150
track_window = (c,r,w,h)

while(1):
    # take first frame of the video
    ret,frame = cap.read()
    if ret == True:
        cv2.rectangle(frame, (c,r), (c+w,r+h), 255, 2)
        cv2.imshow('img1', frame)

        if cv2.waitKey(1) & 0xff == ord('s'):
            break

# No need for the mask
# mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 180], [0, 180, 0, 180])

cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
cv2.imwrite("meanshift_start.jpg",frame)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0,1],roi_hist,[0,180,0,256],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',frame)

        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
        elif k == ord('s'):
            cv2.imwrite("meanshift_snap.jpg",frame)

    else:
        break

cv2.destroyAllWindows()
cap.release()
