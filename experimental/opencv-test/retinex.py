import numpy as np
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil
import cv2
import scipy
import scipy.ndimage

def apply_retinex(X):
    
    # Find luminance and reflectance
    luminance = scipy.ndimage.filters.gaussian_filter(X, 4)
    log_luminance = np.log1p(luminance)
    log_reflectance = np.log1p(X) - log_luminance
    y = np.exp(log_reflectance + 0.2*log_luminance)
    
    y = np.nan_to_num(y)
    y = y.astype(float) / y.max() * 255
    new_frame = y.astype(np.uint8)

    return new_frame

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (320, 240))

    # Convert the frame to LUV
    luv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
    
    # Take ONLY the L channel and apply retinex on it
    luv_channel = luv_frame[:,:,0]
    luv_channel_fixed = apply_retinex(luv_channel)

    # Reconstruct the LUV, convert back to BGR
    luv_frame[:,:,0] = luv_channel_fixed
    new_frame = cv2.cvtColor(luv_frame, cv2.COLOR_LUV2BGR)

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

