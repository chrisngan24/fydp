import numpy as np
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil
import cv2
import scipy
import scipy.ndimage
import time

def apply_retinex(frame, luminance_weighting):
    
    # Convert the frame to LUV
    luv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
            
    # Take ONLY the L channel and apply retinex on it
    luv_channel = luv_frame[:,:,0]
    
    # Find luminance and reflectance
    luminance = scipy.ndimage.filters.gaussian_filter(luv_channel, 4)
    log_luminance = np.log1p(luminance)
    log_reflectance = np.log1p(luv_channel) - log_luminance
    transformed = np.exp(log_reflectance + luminance_weighting * log_luminance)
    
    transformed = np.nan_to_num(transformed)
    transformed = transformed.astype(float) / transformed.max() * 255
    luv_channel_fixed = transformed.astype(np.uint8)

    # Reconstruct the LUV, convert back to BGR
    luv_frame[:,:,0] = luv_channel_fixed
    new_frame = cv2.cvtColor(luv_frame, cv2.COLOR_LUV2BGR)

    return new_frame

cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (320, 240))
    new_frame = apply_retinex(frame, 0.5)

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

