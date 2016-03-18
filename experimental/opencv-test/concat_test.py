import cv2
import numpy as np

image = cv2.imread("retinex_with_best.jpg")
a = np.zeros((240,120,3), dtype=np.uint8)
b = np.concatenate((a,image), axis=1)
layered = np.concatenate((b,a), axis=1)

default_font = cv2.FONT_HERSHEY_SIMPLEX

# Frame is now 240 rows by 560 columns
cv2.putText(img = layered, 
            text = "Head Turn", 
            org = (10,20), 
            fontFace = default_font, 
            fontScale = 0.5, 
            color = (255,255,255), 
            thickness = 1)

cv2.putText(img = layered, 
            text = "Head Turn", 
            org = (445,20), 
            fontFace = default_font, 
            fontScale = 0.5, 
            color = (255,255,255), 
            thickness = 1)

cv2.putText(img = layered, 
            text = "Lane Change", 
            org = (10,140), 
            fontFace = default_font, 
            fontScale = 0.5, 
            color = (255,255,255), 
            thickness = 1)

cv2.putText(img = layered, 
            text = "Lane Change", 
            org = (445,140), 
            fontFace = default_font, 
            fontScale = 0.5, 
            color = (255,255,255), 
            thickness = 1)

# Add sentiments
cv2.putText(img = layered, 
            text = "Good!", 
            org = (15,60), 
            fontFace = default_font, 
            fontScale = 0.75, 
            color = (1,255,1), 
            thickness = 2)

cv2.putText(img = layered, 
            text = "Good!", 
            org = (15,180), 
            fontFace = default_font, 
            fontScale = 0.75, 
            color = (1,255,1), 
            thickness = 2)

cv2.putText(img = layered, 
            text = "Bad!", 
            org = (450,60), 
            fontFace = default_font, 
            fontScale = 0.75, 
            color = (1,1,255), 
            thickness = 2)

cv2.putText(img = layered, 
            text = "Bad!", 
            org = (450,180), 
            fontFace = default_font, 
            fontScale = 0.75, 
            color = (1,1,255), 
            thickness = 2)

print layered.shape
cv2.imshow("frame", layered)
cv2.waitKey(0)