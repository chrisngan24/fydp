import numpy as np
import cv2
import klt
import time
import sys
import pandas as pd
import math

video_name = sys.argv[1]
last_known_good = sys.argv[2]

print "Reading: " + video_name + " to compare with " + last_known_good

cap = cv2.VideoCapture(video_name)
old_frame = []
p0_nose = []
df = pd.DataFrame.from_csv(last_known_good)
idx = 0;
detection_errors = 0
mse = 0

while (cap.isOpened()):

    try:
        (row, old_frame, p0_nose) = klt.getOneEvent(cap, old_frame, p0_nose)
        test_row = df.loc[idx,:]     
        idx += 1

        if (not row.get('isFrontFace') == test_row.isFrontFace):
            detection_errors += 1
    
        xdiff = abs(row.get('noseX') - test_row.noseX)
        ydiff = abs(row.get('noseY') - test_row.noseY)

        mse += math.sqrt(math.pow(xdiff, 2) + math.pow(ydiff, 2))

    except:
        print "Unexpected Error: ", sys.exc_info()[0]
        print "Terminating..."
        cv2.destroyAllWindows()
        cap.release()

# Find average amount of error
mse = mse / (idx - detection_errors)

# Release everything if job is finished
print "Num Detection Errors: " + str(detection_errors) + " out of " + str(idx)
print "MSE " + str(mse)

cap.release()
cv2.destroyAllWindows()

print "Done!"
