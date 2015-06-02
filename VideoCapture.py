__author__ = 'Jerry'
import numpy as np
import cv2
import contours
import handDetection
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
        if type(frame)==type(None):
            continue
        # Our operations on the frame come here
        gray = cv2.blur(frame, (3, 3))
        
        # Process the image and apply skin detection. Get the 100*100 cropped image
        try:
            if type(frame)!=type(None):
                filtered_pic, small =handDetection.detect_hand(frame)
                # filtered_pic=handDetection.detect_face(filtered_pic)
                # filtered_pic=handDetection.detect_eye(filtered_pic)

        except:
            print('no hand found')
        # try:
        #     filtered_pic=contours.imageProcessingForVideos(frame,startTime+str(file_index),saveImages)
        # except:
        #     print 'error'
        # Display the resulting frame
        cv2.imshow('frame',gray)
        cv2.imshow("BW", filtered_pic)
        cv2.imshow("small", small)
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
