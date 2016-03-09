import numpy as np
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil
import cv2
import scipy
import scipy.ndimage

def apply_retinex(X):
    
    # Find luminance and reflectance
    luminance = scipy.ndimage.filters.gaussian_filter(X, 9)
    log_luminance = np.log1p(luminance)
    log_reflectance = np.log1p(X) - log_luminance
    y = np.exp(0.9*log_reflectance + 0.1*log_luminance)
    
    y = np.nan_to_num(y)
    y = y.astype(float) / y.max() * 255
    new_frame = y.astype(int)

    cv2.imwrite('sample.jpg', new_frame)
    new_frame = cv2.imread('sample.jpg')

    return new_frame

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (320, 240))
    new_frame = apply_retinex(frame)

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

