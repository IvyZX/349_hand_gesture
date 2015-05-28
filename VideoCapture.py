__author__ = 'Jerry'
import numpy as np
import cv2
import contours,handDetection
import os,datetime

def main(saveImages=False):
    dir = os.path.dirname(__file__)
    # thresh = 100
    # This will make a folder for the images taken from video. The images will be labeled starting from 1.
    startTime=datetime.datetime.now().strftime("%B %d %Y %H %M %S")+'//'
    if saveImages==True and not os.path.exists(dir+'//video_images//'+startTime):
        os.makedirs(dir+'//video_images//'+startTime)
    cap = cv2.VideoCapture(0)
    cap.set(3,100)
    cap.set(4,100)
    file_index=0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame==None:
            continue
        # Our operations on the frame come here
        gray = cv2.blur(frame, (3, 3))

        '''
        blank_pic = np.zeros((len(gray), len(gray[0]), 3), np.uint8)
        canny = cv2.Canny(gray, thresh, thresh*2, 3)
        result = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, (0, 0))
        contours = result[0]
        hierarchy = result[1]
        for i in range(len(contours)):
            cv2.drawContours(blank_pic, contours, i, (0, 0, 255), 1, 8, hierarchy, 0)
        '''
        #gray = cv2.blur(frame, (3, 3))
        # Process the image and apply skin detection. Get the 100*100 cropped image
        try:
            if frame!=None:
                filtered_pic=handDetection.detect_hand(frame)
                filtered_pic=handDetection.detect_face(filtered_pic)
                filtered_pic=handDetection.detect_eye(filtered_pic)

        except:
            pass
        # try:
        #     filtered_pic=contours.imageProcessingForVideos(frame,startTime+str(file_index),saveImages)
        # except:
        #     print 'error'
        # Display the resulting frame
        cv2.imshow('frame',gray)
        cv2.imshow("BW", filtered_pic)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        file_index+=1
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# If you would like to record the images, change the input to True
# Note: It's not recording the images yet because I've commented out the recording part.
# I am testing the new function to capture hand/face/eyes
main(False)