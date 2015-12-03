import numpy as np
import cv2
import time
import sys
import pandas as pd
import math

FRAME_RESIZE = (320,240)
FOURCC = cv2.cv.CV_FOURCC(*'XVID')
FRAME_RATE = 20

video_name = sys.argv[1]
events = sys.argv[2]

print "Annotating: " + video_name + " with " + events

cap = cv2.VideoCapture(video_name)
old_frame = []
p0_nose = []
df = pd.DataFrame.from_csv(events)
out = cv2.VideoWriter(filename = 'annotated.avi', fourcc = FOURCC, fps = FRAME_RATE, frameSize = FRAME_RESIZE)

event_start_idx = 20
event_end_idx = 70
idx = 0

while (cap.isOpened()):

    (ret, frame) = cap.read()
    if ret==True:
        
        idx += 1
        if (idx > event_start_idx and idx < event_end_idx):
            cv2.rectangle(frame,(20,20),(300,220),(0,255,0), 2)
        
        cv2.imshow('video', frame)
        out.write(frame)
        cv2.waitKey(20)

    else:
        break
        
out.release()
cap.release()
cv2.destroyAllWindows()

print "Done!"
