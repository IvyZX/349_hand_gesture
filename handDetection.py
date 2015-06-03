__author__ = 'Ivy'
import cv2
import numpy as np
import os, math
import contours

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
    handCascade = cv2.CascadeClassifier('Hand.Cascade.1.xml')
    # Apply the cascade to detect the hand. (Don't ask me in detail about how this works. lol)
    hands = handCascade.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(50, 50),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    # Draw a rectangle around the hands
    for (x, y, w, h) in hands[:1]:
        saved = chopImage(image, (x, y, w, h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # saved = image[y:(y + h), x:(x + w)]
    # cv2.imshow("Faces found", image)
    #cv2.waitKey(0)
    if len(hands) == 0:
        return image, None
    return image, saved


def chopImage(image, chop_size):
    if chop_size:
        x, y, w, h = chop_size
        margin_w, margin_h = w/4, h/4
        max_x, max_y = len(image[0]), len(image)
        image = image[max(0, (y-margin_h)):min(max_y, (y+h+margin_h)),
                max(0, (x-margin_w)):min(max_x, (x+w+margin_w))]
    small_gray = cv2.cvtColor(cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY), cv2.cv.CV_GRAY2BGR)
    cont = contours.findCannyContour(small_gray)
    if (cont != []):
        img_area = (y+h+2*margin_h)*(x+w+2*margin_w)
        area = cv2.contourArea(cont[0])
        ratio = area/float(img_area)
        if (ratio > 0.1):
            # filtered = cv2.fillPoly(small_gray, cont, (255, 255, 255))
            filtered = np.copy(small_gray)
            # filtered = contours.formImage(filtered, [], cont)
            cv2.drawContours(filtered, cont, -1, (255, 255, 255), -1, 8)
            filtered = cv2.cvtColor(filtered, cv2.cv.CV_BGR2GRAY)
            for row in range(len(filtered)):
                for col in range(len(filtered[0])):
                    if (filtered[row][col] != 255):
                        filtered[row][col] = 0
            contours.resizeByFilter(filtered, small_gray)
            return cv2.resize(small_gray, (100, 100))
    filtered = contours.filterImage(image)
    return None
    # small = contours.resizeByFilter(filtered, small_gray)
    # return small






# Similar function but used a different cascade file
def detect_face(image):
    grayscale = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = faceCascade.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(50, 50),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow("Faces found", image)
    #cv2.waitKey(0)
    return image


# Similar function but used a different cascade file
def detect_eye(image):
    grayscale = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
    eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    eyes = eyeCascade.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(50, 50),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cv2.imshow("eyes found", image)
    #cv2.waitKey(0)
    return image
