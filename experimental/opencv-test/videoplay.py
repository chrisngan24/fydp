import numpy as np
import cv2

cap = cv2.VideoCapture('drivelog_latest.avi')

while(cap.isOpened()):
    
    (ret, frame) = cap.read()
    if ret==True:

        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
