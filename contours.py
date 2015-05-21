__author__ = 'Ivy'
import cv2
import numpy as np
import os, math
dir = os.path.dirname(__file__)

# blur and filter the image before finding contours
def filterImage(src):
    # soften image and remove noise
    pic = cv2.blur(src, (3, 3))
    pic = cv2.GaussianBlur(pic, (11, 11), 0)
    pic = cv2.medianBlur(pic, 11)
    pic = cv2.cvtColor(pic, cv2.cv.CV_BGR2HSV)
    # apply threshold on hsv values to detect skin color
    pic = cv2.inRange(pic, (0, 40, 80, 255), (50, 225, 255, 255))
    # apply morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    pic = cv2.morphologyEx(pic, cv2.MORPH_OPEN, kernel)
    pic = cv2.GaussianBlur(pic, (3, 3), 0)
    return pic


# return the contour and hierarchy of the canny contour
def findCannyContour(pic):
    thresh = 100
    canny = cv2.Canny(pic, thresh, thresh*2, 3)
    result = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, (0, 0))
    contours = result[0]
    hierarchy = result[1]
    # find the contour with greatest area
    max_area = -1
    contour = None
    for c in contours:
        area = cv2.contourArea(c)
        if (area > max_area):
            max_area = area
            contour = c
    # approximate contour with poly-line
    if contour is None:
        return []
    contour = cv2.approxPolyDP(contour, 2, False)
    return [contour]

# return points of the convex hull
def findConvexHull(contours):
    hull=cv2.convexHull(contours[0],returnPoints=False)
    if hull is None:
        return
    defects=cv2.convexityDefects(contours[0],hull)
    if defects is None:
        return
    convex_pts = [None] * defects.shape[0]
    #find the convex points and the lines connecting the hand's convex hull points
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(contours[0][s][0])
        end = tuple(contours[0][e][0])
        far = tuple(contours[0][f][0])
        convex_pts[i] = [start, end, far]
    return convex_pts

# given the contours and convex points, draw them on the image and return it
def formImage(blank_pic, convex_pts, contours):
    for cpt in convex_pts:
        start, end, far = cpt
        cv2.line(blank_pic, start, end, (0, 255, 0), 2)
        cv2.circle(blank_pic, far, 5, (255, 0, 0), -1)
    for i in range(len(contours)):
        cv2.drawContours(blank_pic, contours, i, (0, 0, 255), 1, 8)
    return blank_pic

# form a pop-up window that shows the images
def popupImage(pic):
    cv2.imshow("Result", pic)
    cv2.waitKey()

# The main function that accepts an image and returns a gray image with convex points
def main(src, popup):
    blank_pic = np.zeros((len(src), len(src[0]), 3), np.uint8)
    gray_pic = cv2.cvtColor(cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY), cv2.cv.CV_GRAY2BGR)
    
    #filtered = cv2.blur(src, (3, 3))
    filtered = filterImage(src)
    
    contours = findCannyContour(filtered)
    if (contours == []):
        print "No contours found"
        return gray_pic
    
    convex_pts = findConvexHull(contours)
    if convex_pts is None:
        print "No convex points found"
        return gray_pic
    
    gray_pic = formImage(gray_pic, convex_pts, contours)
    if popup:
        popupImage(gray_pic)
    return gray_pic

#main(cv2.imread(dir+"//hand pics//10.jpg"), True)


'''

    # If you want to draw hull, returnPoints has to be True. But defects needs it to be False.
    # It's not so useful anyway
    # for i in range(len(hull)):
    #     cv2.drawContours(blank_pic,hull,i,(0,255,0),1,8,hierarchy,0)

    # Center of mass and average radius. THere should be easier way to do this.
    # (ie. built in functions in cv2)
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
    # cv2.circle(blank_pic,(int(x_com),int(y_com)),int(radius),(0,0,255),5)
    '''
