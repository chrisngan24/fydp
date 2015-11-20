import numpy as np
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil
import cv2

cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    new_frame = to_pil(cca.stretch(from_pil(frame)))

    # Display the resulting frame
    cv2.imshow('frame',new_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

