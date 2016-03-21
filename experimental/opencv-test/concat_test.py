import cv2
import numpy as np

image = cv2.imread("retinex_with_best.jpg")
image = cv2.resize(image, (480,360))
a = np.ones((360,180,3), dtype=np.uint8)
a = a * 255
b = np.concatenate((a,image), axis=1)
layered = np.concatenate((b,a), axis=1)

default_font = cv2.FONT_HERSHEY_SIMPLEX

# Frame is now 240 rows by 560 columns
cv2.putText(img = layered, 
            text = "Right Head", 
            org = (2,30), 
            fontFace = default_font, 
            fontScale = 1, 
            color = (255,255,255), 
            thickness = 1)

cv2.putText(img = layered, 
            text = "Turn", 
            org = (50,60), 
            fontFace = default_font, 
            fontScale = 1, 
            color = (255,255,255), 
            thickness = 1)

cv2.putText(img = layered, 
            text = "Left Head", 
            org = (675,30), 
            fontFace = default_font, 
            fontScale = 1, 
            color = (255,255,255), 
            thickness = 1)

cv2.putText(img = layered, 
            text = "Turn", 
            org = (715,60), 
            fontFace = default_font, 
            fontScale = 1, 
            color = (255,255,255), 
            thickness = 1)

cv2.putText(img = layered, 
            text = "Right Lane", 
            org = (2,200), 
            fontFace = default_font, 
            fontScale = 1, 
            color = (255,255,255), 
            thickness = 1)

cv2.putText(img = layered, 
            text = "Change", 
            org = (30,235), 
            fontFace = default_font, 
            fontScale = 1, 
            color = (255,255,255), 
            thickness = 1)

cv2.putText(img = layered, 
            text = "Left Lane", 
            org = (675,200), 
            fontFace = default_font, 
            fontScale = 1, 
            color = (255,255,255), 
            thickness = 1)

cv2.putText(img = layered, 
            text = "Change", 
            org = (690,235), 
            fontFace = default_font, 
            fontScale = 1, 
            color = (255,255,255), 
            thickness = 1)

# Add sentiments
cv2.putText(img = layered, 
            text = "Good!", 
            org = (20,120), 
            fontFace = default_font, 
            fontScale = 1.5, 
            color = (1,255,1), 
            thickness = 2)

cv2.putText(img = layered, 
            text = "Good!", 
            org = (20,290), 
            fontFace = default_font, 
            fontScale = 1.5, 
            color = (1,255,1), 
            thickness = 2)

cv2.putText(img = layered, 
            text = "Bad!", 
            org = (700,120), 
            fontFace = default_font, 
            fontScale = 1.5, 
            color = (1,1,255), 
            thickness = 2)

cv2.putText(img = layered, 
            text = "Bad!", 
            org = (700,290), 
            fontFace = default_font, 
            fontScale = 1.5, 
            color = (1,1,255), 
            thickness = 2)

print layered.shape
cv2.imshow("frame", layered)
cv2.waitKey(0)