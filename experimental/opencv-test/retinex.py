import numpy as np
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    new_frame = cca.stretch(frame)

    # Display the resulting frame
    cv2.imshow('frame',new_frame)
    cv2.imshow('frame_without',frame)
    
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break
    elif k == ord('s'):
        cv2.imwrite('retinex_with.jpg', new_frame)
        cv2.imwrite('retinex_without.jpg', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

