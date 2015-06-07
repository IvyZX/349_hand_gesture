__author__ = 'Ivy'
import cv2
import numpy as np
import contours
import os, random, sys, collections, datetime, math

dir = os.path.dirname(__file__)

# Helper functions
def most_common(lst):
    return max(set(lst), key=lst.count)


# Suppose the data set is a list of data
# in which each data is [image, gesture_id]

# implement the k nearest neighbor
# basically, find the existed picture with the highest similarity
def findNearestGesture(data_set, hand_pic, k=2):
    # k=int(math.sqrt(len(data_set)))
    sim_gesture_array = []
    if hand_pic.shape!=(100,100):
        hand_pic = cv2.cvtColor(hand_pic, cv2.cv.CV_BGR2GRAY)
    for data in data_set:
        sim = computeDifference(data[0], hand_pic)
        sim_gesture_array.append([sim, data[1]])
    # Note: The sorting here is smaller numbers first.
    # So if you want to test some other methods in computeDifference function, you may change reverse=False to True
    if k==1:
        [max_sim,gesture_id]=max(sim_gesture_array,key=lambda x: x[0])
    else:
        sim_gesture_array.sort(key=lambda x: x[0], reverse=False)
        max_sim = sum(v[0] for v in sim_gesture_array[:k]) / float(k)
        gesture_id = most_common([v[1] for v in sim_gesture_array[:k]])
    #print sim_gesture_array[:k]
    return gesture_id, max_sim


# return a similarity rating that ranges in between 0-1
# is the percentage that the common pixel occupies the whole hand
# all non-black pixels are considered the same here, assuming it's only black&white
def computeDifference(data_pic, hand_pic):
    # This method is 150x faster (Did not test the accuracy)
    diff = np.subtract(data_pic, hand_pic)
    diff = np.square(diff)
    return float(diff.sum())

    '''
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
    '''

# Converts all larger-than-0 pixels to 1.
def convertImgToBinary(pic):
    for i in range(len(pic)):
        for j in range(len(pic)):
            pic[i][j]=int(bool(pic[i][j]))

# The main training/outputting data set function.
# It automatically reads directories in video_images and assume pictures
# in the same folder belongs to same category
# It returns a list of data in which each data is [image, gesture_id]
def mainTrain():  # (file_dir=dir, file_list=None, ges_id_list=None):
    # ges_folder_list includes all sub-folders in video_images folder(include the video_images folder itself)
    ges_folder_list = [x[0] for x in os.walk(dir + '//video_images//')]
    # file_list includes all files/direct sub-folders under the folders in ges_folder_list.
    # Therefore its first element would be a list of folder names under video_images which we take as the category names
    file_list = [[file for file in os.listdir(subdir)] for subdir in ges_folder_list]
    data_set = []
    if (len(file_list) != len(ges_folder_list)):
        print "error"
    else:
        for category in range(len(file_list[0])):
            for i in file_list[category + 1]:
                pic = cv2.imread(ges_folder_list[category + 1] + '//' + i)
                # pic = contours.main(pic, False)
                data_set.append([pic, file_list[0][category]])
    for data in data_set:
        data[0] = cv2.cvtColor(data[0], cv2.cv.CV_BGR2GRAY)
        convertImgToBinary(data[0])

    return data_set


# This is the ten fold cross validation function
# It split the data into training and validation set.
# It will automatically call the findNearestGesture and determine the performance of our algorithm
# Input: precision recall file name, and true-false table file name
# Output: nothing, but writes the results into two files.
def tenFoldCrossValidation(neighbor_num=3, precisionRecallFileName='precisionRecallFile.csv',
                           trueFalseTableFileName='trueFalseTable.csv'):
    # Initialize Precisions and Recall values
    precision = 0.0
    recall = 0.0
    now = datetime.datetime.now()
    with open(trueFalseTableFileName, 'a') as trueFalseTableFile:
        trueFalseTableFile.write('Training started on: ' + str(now) + '\n')
        trueFalseTableFile.write('Trial Number,Gesture,True Positives,False Positives,False Negatives\n')
    for i in range(0, 10):

        data_set = mainTrain()
        # shuffle the list (in the same manner every time) so that there's an equal chance to get positives/negatives
        random.seed(0)
        random.shuffle(data_set)
        training_data_set = data_set[:i * len(data_set) / 10] + data_set[(i + 1) * len(data_set) / 10:]
        validation_data_set = data_set[i * len(data_set) / 10:(i + 1) * len(data_set) / 10]
        # Initialize counters to calculate precision recall
        falsePos = collections.defaultdict(float);
        falseNeg = collections.defaultdict(float);
        truePos = collections.defaultdict(float);

        # Validating
        counter = 0.0
        for data in validation_data_set:
            gesture_id, max_sim = findNearestGesture(training_data_set, data[0], neighbor_num)
            if gesture_id == data[1]:
                truePos[data[1]] += 1
            else:
                falsePos[gesture_id] += 1
                falseNeg[data[1]] += 1
            counter += 1
            if counter % 10 == 0:
                sys.stdout.write('\rtrial ' + str(i) + ':' + str(counter / len(validation_data_set) * 100) + '% Done')
        sumTruePos = sum(truePos.values());
        sumFalsePos = sum(falsePos.values());
        sumFalseNeg = sum(falseNeg.values())
        precision += (sumTruePos / (sumTruePos + sumFalsePos))
        recall += (sumTruePos / (sumTruePos + sumFalseNeg))
        with open(trueFalseTableFileName, 'a') as trueFalseTableFile:
            for key in truePos.keys():
                trueFalseTableFile.write(
                    'trial ' + str(i) + ',' + str(key) + ',' + str(truePos[key]) + ',' + str(falsePos[key]) + ',' + str(
                        falseNeg[key]) + '\n')
                #print (str(truePos)+','+str(falsePos))
    precision /= 10
    recall /= 10
    f1 = 2 * precision * recall / (precision + recall)
    with open(precisionRecallFileName, 'a') as precisionRecallFile:
        precisionRecallFile.write('Training started on: ' + str(now) + '\n')
        precisionRecallFile.write('Precision,Recall,F1\n')
        precisionRecallFile.write(str(precision) + ',' + str(recall) + ',' + str(f1) + '\n')
    print ('Precision: ' + str(precision) + ', Recall: ' + str(recall) + ', F1: ' + str(f1) + '\n')
    return

def crossValidationAcrossDifferentPeople(neighbor_num=2, precisionRecallFileName='precisionRecallFile.csv',
                           trueFalseTableFileName='trueFalseTable.csv'):
    # Initialize Precisions and Recall values
    precision = 0.0
    recall = 0.0
    now = datetime.datetime.now()
    with open(trueFalseTableFileName, 'a') as trueFalseTableFile:
        trueFalseTableFile.write('Training started on: ' + str(now) + '\n')
        trueFalseTableFile.write('Trial Number,Gesture,True Positives,False Positives,False Negatives\n')
    for i in range(3, 10):

        data_set = mainTrain()
        # shuffle the list (in the same manner every time) so that there's an equal chance to get positives/negatives
        random.seed(0)
        random.shuffle(data_set)
        training_data_set=[]
        validation_data_set=[]
        for data in data_set:
            if data[1][-1]==str(i):
                validation_data_set.append(data)
            else:
                training_data_set.append(data)
        # Initialize counters to calculate precision recall
        falsePos = collections.defaultdict(float);
        falseNeg = collections.defaultdict(float);
        truePos = collections.defaultdict(float);

        # Validating
        counter = 0.0
        for data in validation_data_set:
            gesture_id, max_sim = findNearestGesture(training_data_set, data[0], neighbor_num)
            if gesture_id[:-1] == data[1][:-1]:
                truePos[data[1][:-1]] += 1
            else:
                falsePos[gesture_id[:-1]] += 1
                falseNeg[data[1][:-1]] += 1
            counter += 1
            if counter % 10 == 0:
                sys.stdout.write('\rtrial ' + str(i) + ':' + str(counter / len(validation_data_set) * 100) + '% Done')
        sumTruePos = sum(truePos.values());
        sumFalsePos = sum(falsePos.values());
        sumFalseNeg = sum(falseNeg.values())
        precision += (sumTruePos / (sumTruePos + sumFalsePos))
        recall += (sumTruePos / (sumTruePos + sumFalseNeg))
        with open(trueFalseTableFileName, 'a') as trueFalseTableFile:
            for key in truePos.keys():
                trueFalseTableFile.write(
                    'trial ' + str(i) + ',' + str(key) + ',' + str(truePos[key]) + ',' + str(falsePos[key]) + ',' + str(
                        falseNeg[key]) + '\n')
                #print (str(truePos)+','+str(falsePos))
    precision /= 7
    recall /= 7
    f1 = 2 * precision * recall / (precision + recall)
    with open(precisionRecallFileName, 'a') as precisionRecallFile:
        precisionRecallFile.write('Training started on: ' + str(now) + '\n')
        precisionRecallFile.write('Precision,Recall,F1\n')
        precisionRecallFile.write(str(precision) + ',' + str(recall) + ',' + str(f1) + '\n')
    print ('Precision: ' + str(precision) + ', Recall: ' + str(recall) + ', F1: ' + str(f1) + '\n')
    return

def outputArff():
    reduced_dimension=25
    data_set = mainTrain()
    random.seed(0)
    random.shuffle(data_set)
    with open('imageData.arff','w') as f:
        f.write('  % 1. Title: EECS 349 Final Project Data\n')
        f.write('   @RELATION EECS-349-Final-Project-Data\n')

        for i in range(reduced_dimension**2):
            f.write('   @ATTRIBUTE '+str(i)+' NUMERIC\n')
        classes=['A','B','C','One','Two','Three','Four','Five']
        f.write('@ATTRIBUTE class {A,B,C,One,Two,Three,Four,Five}\n')
        f.write('   @DATA\n')

        for data in data_set:
            img=cv2.resize(data[0], (reduced_dimension, reduced_dimension))
            for i in range(reduced_dimension):
                for j in range(reduced_dimension):
                    f.write(str(data[0][i][j])+',')
            f.write(str(data[1][:-1])+'\n')
    print 'Successfully written all image data into imageData.arff'



if __name__ == "__main__":
    startTime = datetime.datetime.now()
    # If you want to do a single image:
    # data_set = mainTrain()
    # test_pic = cv2.imread(dir+'//video_images//C//11.png')
    # print(findNearestGesture(data_set, test_pic))

    # If you want to do the ten fold cross validation. (Which takes super long for our KNN algorithm)
    #tenFoldCrossValidation()
    crossValidationAcrossDifferentPeople(neighbor_num=2)
    #outputArff()
    endTime = datetime.datetime.now()
    print 'Total time: ' + str(endTime - startTime)

# Changelog
# Jerry 05/25/2015
# Major changes made:
# 1. Added the ten fold cross validation function. Now we can evaluate how good our models are. (Or how bad I filmed my hand)
# 2. Changed the mainTrain() function so now it automatically looks for categories in the video_images folder.
# 3. Changed the findNearestGesture() so that now it could do KNN instead of one nearest neighbor
# 4. Changed Compute difference so that it now runs 150 times faster than the old model. It now computes difference between every pixel and sum them. This may affect the accuracy.
# Other things to mention: For now the precision and recall have the same value. This is mathematically correct (I think). It just happens because we have more than 2 categories.
