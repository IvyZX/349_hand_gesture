__author__ = 'Jerry'
import numpy as np
import cv2
thresh = 100
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #print frame
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(frame, (3, 3))

    blank_pic = np.zeros((len(gray), len(gray[0]), 3), np.uint8)

    canny = cv2.Canny(gray, thresh, thresh*2, 3)
    result = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, (0, 0))
    contours = result[0]
    hierarchy = result[1]
    for i in range(len(contours)):
        cv2.drawContours(blank_pic, contours, i, (0, 0, 255), 1, 8, hierarchy, 0)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    cv2.imshow("Result", blank_pic)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()