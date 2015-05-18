import cv2
import numpy as np
import os, math
dir = os.path.dirname(__file__)

thresh = 100

src = cv2.imread(dir+"\\hand pics\\04.jpg")

#gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = cv2.blur(src, (3, 3))
blank_pic = np.zeros((len(gray), len(gray[0]), 3), np.uint8)

canny = cv2.Canny(gray, thresh, thresh*2, 3)
result = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, (0, 0))

contours = result[0]
hierarchy = result[1]

# Center of mass and average radius. THere should be easier way to do this. (ie. built in functions in cv2)
x_com=0.0
y_com=0.0
for i in range(len(contours[0])):
    x_com+=contours[0][i][0][0]
    y_com+=contours[0][i][0][1]
x_com/=len(contours[0])
y_com/=len(contours[0])
radius=0.0
for i in range(len(contours[0])):
    radius+=math.sqrt((x_com-contours[0][i][0][0])**2+(y_com-contours[0][i][0][1])**2)
radius/=len(contours[0])
# Draw the circle representing the object and its center of mass.
#cv2.circle(blank_pic,(int(x_com),int(y_com)),int(radius),(0,0,255),5)

# Convex hull
hull=cv2.convexHull(contours[0],returnPoints=False)
defects=cv2.convexityDefects(contours[0],hull)

# If you want to draw hull, returnPoints has to be True. But defects needs it to be False. It's not so useful anyway
# for i in range(len(hull)):
#     cv2.drawContours(blank_pic,hull,i,(0,255,0),1,8,hierarchy,0)

#Draw the convex points and the lines connecting the hand's convex hull points
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(contours[0][s][0])
    end = tuple(contours[0][e][0])
    far = tuple(contours[0][f][0])
    cv2.line(blank_pic,start,end,[0,255,0],2)
    cv2.circle(blank_pic,far,5,[0,0,255],-1)

# Draw the actual contours of the hand
for i in range(len(contours)):
    cv2.drawContours(blank_pic, contours, i, (0, 0, 255), 1, 8, hierarchy, 0)

#cv2.imwrite(dir+"\\hand pics\\05-c.jpg", blank_pic)

cv2.imshow("Result", blank_pic)
cv2.waitKey()
