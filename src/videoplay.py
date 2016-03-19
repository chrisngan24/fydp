import cv2
import sys

video_name = sys.argv[1]
cap = cv2.VideoCapture(video_name)
i = 0
while(cap.isOpened()):
    
    (ret, frame) = cap.read()
    if ret==True:

        cv2.imshow('frame',frame)
        k = cv2.waitKey(20) & 0xff
        i += 1
        if k == ord('q'):
            break
        elif k == ord('s'):
            cv2.imwrite("snap.jpg",frame)
    else:
        print i
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
