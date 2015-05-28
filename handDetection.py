__author__ = 'Ivy'
import cv2
import numpy as np
import os, math
dir = os.path.dirname(__file__)

# This is the hand detection function that uses Haar Cascade to detect hand (in good, bright lighting conditions)
# Input: the image matrix
# Output: the image with a square at the hand's position.
# We could change the output of course if we are satisfied with the result
def detect_hand(image):
    # Citation: https://github.com/shantnu/FaceDetect/blob/master/face_detect.py
    # CItation: http://stackoverflow.com/questions/8593091/robust-hand-detection-via-computer-vision
    # Citation: http://nmarkou.blogspot.com/2012/02/haar-xml-file.html
    # Convert to gray picture
    grayscale = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
    # load the cascade
    handCascade=cv2.CascadeClassifier('Hand.Cascade.1.xml')
    # Apply the cascade to detect the hand. (Don't ask me in detail about how this works. lol)
    hands=handCascade.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(50,50),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    # Draw a rectangle around the hands
    for (x, y, w, h) in hands:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #cv2.imshow("Faces found", image)
    #cv2.waitKey(0)
    return image

# Similar function but used a different cascade file
def detect_face(image):
    grayscale = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY) 
    faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') 
    faces=faceCascade.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(50,50),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #cv2.imshow("Faces found", image)
    #cv2.waitKey(0)
    return image

# Similar function but used a different cascade file
def detect_eye(image):
    grayscale = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY) 
    eyeCascade=cv2.CascadeClassifier('haarcascade_eye.xml') 
    eyes=eyeCascade.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(50,50),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    #cv2.imshow("eyes found", image)
    #cv2.waitKey(0)
    return image


# blur and filter the image before finding contours
def filterImage(src):
    # soften image and remove noise
    pic = cv2.blur(src, (3, 3))
    pic = cv2.GaussianBlur(pic, (11, 11), 0)
    pic = cv2.medianBlur(pic, 11)
    #print pic
    pic = cv2.cvtColor(pic, cv2.cv.CV_BGR2HSV)
    # apply threshold on hsv values to detect skin color
    #print pic
    # In opencv 8-bit images. h<-h/2 (to fit 255), s<-255s, v<-255v. Originally 0<=h<360, 0<=s<=1, 0<=v<=1
    # From preprcessing.pdf page 7
    # V>=40
    # 0.2<=S<=0.6
    # 0<=H<=25 or 335<=H<=360

    lowerRange=np.array([0,51,40])
    upperRange=np.array([25,153,255])
    #pic = cv2.inRange(pic, (0, 40, 80, 255), (50, 225, 255, 255))
    pic = cv2.inRange(pic, lowerRange, upperRange)
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

# This function automatically detects the skin and resize the window to the size of the hand detected
def applyFilter(filter,pic):
    rowMin=len(pic);rowMax=0;columnMin=len(pic[0]);columnMax=0
    for row in range(len(pic)):
        for column in range(len(pic[row])):
            if filter[row][column]==0:
                pic[row][column]=[0,0,0]
            else:
                rowMin=min(rowMin,row)
                rowMax=max(rowMax,row)
                columnMin=min(columnMin,column)
                columnMax=max(columnMax,column)
    if (rowMax>rowMin and columnMax>columnMin):
        pic=pic[rowMin:rowMax,columnMin:columnMax]
    try:
        pic=cv2.resize(pic,(100,100))
    except:
        pass
    return pic


# This function will rotate the whole picture by an angle
# may use it later
def rotateImage(pic, angle):
    pt = (len(pic)*0.5, len(pic[0])*0.5)
    rot = cv2.getRotationMatrix2D(pt, angle, 1.0)
    new = cv2.warpAffline(pic, r, (1.7*len(pic), 1.7*len(pic[0])))
    return new




# This function will be called in VideoCapture.py.
# Inputs: src<- the image, file_index<-the index of the image, or any file name you would like to name the image.
# saveImages<- boolean variable. The function will save the images if it's true.
# Outputs: cropped 100*100 image of the hand.
def imageProcessingForVideos(src,file_index,saveImages):
    gray_pic = cv2.cvtColor(cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY), cv2.cv.CV_GRAY2BGR)
    # Apply skin filter on the source image
    filtered = filterImage(src)
    # Apply the filter on the gray image and automatically crop the image so that the hand could takes up the whole screen
    gray_pic=applyFilter(filtered,gray_pic)
    if saveImages==True:
        cv2.imwrite( dir+"//video_images//"+str(file_index)+".png", gray_pic );
    #Saves the processed and filtered gray picture.

    return gray_pic

# The main function that accepts an image and returns a gray image with convex points
def main(src, popup):
    blank_pic = np.zeros((len(src), len(src[0]), 3), np.uint8)
    gray_pic = cv2.cvtColor(cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY), cv2.cv.CV_GRAY2BGR)
    color_pic=src
    filtered = filterImage(src)
    gray_pic=applyFilter(filtered,gray_pic)
    # contours = findCannyContour(gray_pic)
    # if (contours == []):
    #     print "No contours found"
    #     return gray_pic
    #
    # convex_pts = findConvexHull(contours)
    # if convex_pts is None:
    #     print "No convex points found"
    #     return gray_pic
    #
    # gray_pic = formImage(gray_pic, convex_pts, contours)
    if popup:
        popupImage(gray_pic)
    return gray_pic

#main(cv2.imread(dir+"//shp_marcel_train//Five//Five-train300.jpg"), True)
detect_hand(cv2.imread(dir+"//video_images//A//11.png"))

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
