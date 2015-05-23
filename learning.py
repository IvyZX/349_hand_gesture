__author__ = 'Ivy'
import cv2
import numpy as np
import contours
import os
dir = os.path.dirname(__file__)

# Suppose the data set is a list of data
# in which each data is [image, gesture_id]

# implement the nearest neighbor for now
def findNearestGesture(data_set, hand_pic):
    max_sim = 0
    gesture_id = -1
    for data in data_set:
        sim = computeDifference(data[0], hand_pic)
        if (sim > max_sim):
            max_sim = sim
            gesture_id = data[1]
    return gesture_id, max_sim

# return a difference that ranges in between 0-1
# is the percentage that the common pixel occupies the whole hand
def computeDifference(data_pic, hand_pic):
    total = 0
    common = 0
    black = np.zeros((3), np.uint8)
    for row in range(len(data_pic)):
        for col in range(len(data_pic[0])):
            #print type(hand_pic[row][col])
            if not np.array_equal(hand_pic[row][col], black):
                total += 1
                if not np.array_equal(data_pic[row][col], black):
                    common += 1
    return float(common)/float(total)

def mainTrain(): #(file_dir, file_list, ges_id_list):
    # make it up for now
    file_dir = dir+'//video_images//May 22 2015 15 00 07//'
    file_list = map(lambda n: str(n)+'.png', range(27))
    ges_id_list = [0]*8 + [1]*6 + [2]*6 + [3]*7
    data_set = []
    if (len(file_list) != len(ges_id_list)):
        print "error"
    else:
        for i in range(len(file_list)):
            pic = cv2.imread(file_dir+file_list[i])
            #pic = contours.main(pic, False)
            data_set.append([pic, ges_id_list[i]])
    return data_set

data_set = mainTrain()
test_pic = cv2.imread(dir+'//video_images//May 22 2015 15 00 07//30.png')
print(findNearestGesture(data_set, test_pic))





