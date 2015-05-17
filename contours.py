import cv2
import numpy as np
import os
dir = os.path.dirname(__file__)

thresh = 100

src = cv2.imread(dir+"\\hand pics\\03.jpg")

#gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = cv2.blur(src, (3, 3))
blank_pic = np.zeros((len(gray), len(gray[0]), 3), np.uint8)

canny = cv2.Canny(gray, thresh, thresh*2, 3)
result = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, (0, 0))
contours = result[0]
hierarchy = result[1]

for i in range(len(contours)):
    cv2.drawContours(blank_pic, contours, i, (0, 0, 255), 1, 8, hierarchy, 0)

#cv2.imwrite(dir+"\\hand pics\\05-c.jpg", blank_pic)

cv2.imshow("Result", blank_pic)
cv2.waitKey()
