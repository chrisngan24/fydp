import cv2
import numpy as np

image = cv2.imread("retinex_with_best.jpg")
a = np.zeros((240,120,3), dtype=np.uint8)
b = np.concatenate((a,image), axis=1)
layered = np.concatenate((b,a), axis=1)

default_font = cv2.FONT_HERSHEY_SIMPLEX

# Frame is now 240 rows by 560 columns
cv2.putText(img = layered, 
            text = "Right Head", 
            org = (10,20), 
            fontFace = default_font, 
            fontScale = 0.5, 
            color = (255,255,255), 
            thickness = 1)

cv2.putText(img = layered, 
            text = "Turn", 
            org = (25,40), 
            fontFace = default_font, 
            fontScale = 0.5, 
            color = (255,255,255), 
            thickness = 1)

cv2.putText(img = layered, 
            text = "Left Head", 
            org = (450,20), 
            fontFace = default_font, 
            fontScale = 0.5, 
            color = (255,255,255), 
            thickness = 1)

cv2.putText(img = layered, 
            text = "Turn", 
            org = (465,40), 
            fontFace = default_font, 
            fontScale = 0.5, 
            color = (255,255,255), 
            thickness = 1)

cv2.putText(img = layered, 
            text = "Right Lane", 
            org = (10,140), 
            fontFace = default_font, 
            fontScale = 0.5, 
            color = (255,255,255), 
            thickness = 1)

cv2.putText(img = layered, 
            text = "Change", 
            org = (25,160), 
            fontFace = default_font, 
            fontScale = 0.5, 
            color = (255,255,255), 
            thickness = 1)

cv2.putText(img = layered, 
            text = "Left Lane", 
            org = (450,140), 
            fontFace = default_font, 
            fontScale = 0.5, 
            color = (255,255,255), 
            thickness = 1)

cv2.putText(img = layered, 
            text = "Change", 
            org = (460,160), 
            fontFace = default_font, 
            fontScale = 0.5, 
            color = (255,255,255), 
            thickness = 1)

# Add sentiments
cv2.putText(img = layered, 
            text = "Good!", 
            org = (15,80), 
            fontFace = default_font, 
            fontScale = 0.75, 
            color = (1,255,1), 
            thickness = 2)

cv2.putText(img = layered, 
            text = "Good!", 
            org = (15,200), 
            fontFace = default_font, 
            fontScale = 0.75, 
            color = (1,255,1), 
            thickness = 2)

cv2.putText(img = layered, 
            text = "Bad!", 
            org = (450,80), 
            fontFace = default_font, 
            fontScale = 0.75, 
            color = (1,1,255), 
            thickness = 2)

cv2.putText(img = layered, 
            text = "Bad!", 
            org = (450,200), 
            fontFace = default_font, 
            fontScale = 0.75, 
            color = (1,1,255), 
            thickness = 2)

print layered.shape
cv2.imshow("frame", layered)
cv2.waitKey(0)