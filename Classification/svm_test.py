from sklearn.svm import SVC


import scipy.io as sio 
import os
import numpy as np
import tensorflow as tf
from random import shuffle
from math import ceil
from tensorflow.models.rnn import rnn, rnn_cell

# File directory
matFileRootDirect = u'/home/peizhen/HON4D_feature_part_10_frame_max'

train_data=[]
test_data=[]
test_label=[]
train_label=[]
actual_train_input_len=[]
actual_test_input_le=[]

clf = SVC()

# Get ".mat" file paths
def getMatFileList():
        matFileList = []
        for i in range(6):
                videoName = 'video%02u'%(i+1)
                matFileName = videoName+'_HON4D_feature.mat'
                matFileList.append(matFileName)
        for i in range(81):
                videoName = 'video%02u'%(i+15)
                matFileName = videoName+'_HON4D_feature.mat'
                matFileList.append(matFileName)
        return matFileList

# Construct training set and test set
def getData(matFileList):
	global train_data
        global train_label
        global test_data
        global test_label
        global actual_train_input_len
        global actual_test_input_len

        allVideoDatas = []
	allVideoLabels = []
	# activities[i] store several ranges, each range corresponds to several rows with label i in AllVideoInfos
        activities = [[],[],[],[],[],[],[]]



        # Iterate all mat files (all video information)
        for i in range(len(matFileList)):
                matFilePth = os.path.join(matFileRootDirect,matFileList[i])
                data = sio.loadmat(matFilePth)
                curVideoData = data['current_video_data']
                features = curVideoData[0][0][0]        # Get all features of current video -> k x 3240
                label = curVideoData[0][0][1][0][0]     # Get label of current video        -> an integer

                # Transfer current video information to several rows of vectors(each vector with at most 25 elements)
                # Each element is a 1 x 3240 vector

                # Range indicates the position of current video information in allVideoDatas
                # Which will be append to activities[label]
                beginIndex = endIndex = len(allVideoDatas)

                # Used for counting how many rows in allVideoDatas are taken by current video
                n_takenRows = 0

                thisRowFeature = []
                origIndex = 0
                while origIndex < len(features):
                        thisRowFeature.append(features[origIndex])
                        if len(thisRowFeature) == 25:
                                allVideoDatas.append(thisRowFeature)
                                thisRowFeature = []
                                n_takenRows += 1
                        origIndex += 4

                # Just for test
                lastRowFeatureNum = len(thisRowFeature)
                # If the length of last row is less than 25
                if len(thisRowFeature) > 0:
                        # If the length is longer than 10, append it to "allVideoDatas" as a new row
                        if len(thisRowFeature) >= 10:
                                n_takenRows += 1
                                allVideoDatas.append(thisRowFeature)
                        # ... shorter than 10, extend it to the last row of "allVideoDatas"
                        else:
                                lastIndex = len(allVideoDatas)-1
                                allVideoDatas[lastIndex].extend(thisRowFeature)
                        thisRowFeature = []

                endIndex = beginIndex + n_takenRows - 1

                # Append the range to corresponding activity
                activities[label].append([beginIndex, endIndex])


                #videoName = "%.*s" % (7,matFileList[i])
                #print "%s  take  %u rows, last row feature number: %u" % (videoName,n_takenRows,lastRowFeatureNum)
                #print(label)
	 # Iterate each activity
        for i in range(len(activities)):
                n_ownedVideo = len(activities[i])       # Number of videos belong to activities[i]
                # Approximately distribute videos to training and test set according to rate 4:1
                # n_toTest + n_toTrain = n_ownedVideo
                n_toTest = int(ceil((float)(n_ownedVideo)/5))
                n_toTrain = n_ownedVideo - n_toTest

                #empty_label = [0]*n_classes
                #empty_label[i] = 1

                # Shuffle elements (ranges) in activities[i] and distribute them to training and test set
                shuffle(activities[i])

                # Add the rows in "allVideoDatas" within the former "n_toTest" ranges in activities[i] to test_data
                for j in range(n_toTest):
                        curRange = activities[i][j]
                        beginIndex = curRange[0]
                        endIndex = curRange[1]
                        test_data.extend(allVideoDatas[beginIndex:endIndex+1])
                        # Add "endIndex-beginIndex+1" labels to the end of test_label, each label is transferred to an array of length 7
                        # in which the element at the "label index" is 1 and others 0
			# Just for svm test, each label is a value but not an array
                        #test_label.extend([empty_label]*(endIndex-beginIndex+1))
			test_label.extend([i]*(endIndex-beginIndex+1))


                # Add the rows in "allVideoDatas" within the later "n_toTrain" ranges in activities[i] to train_data
                offset = n_toTest
                for j in range(n_toTrain):
                        curRange = activities[i][j+offset]
                        beginIndex = curRange[0]
                        endIndex = curRange[1]
                        train_data.extend(allVideoDatas[beginIndex:endIndex+1])
                        # Add "endIndex-beginIndex+1" labels to the end of train_label
                        #train_label.extend([empty_label]*(endIndex-beginIndex+1))
			# Just for svm test, each label is a value but not an array
			train_label.extend([i]*(endIndex-beginIndex+1))

	# Release the memory taken by allVideoDatas
        del allVideoDatas

        # Shuffle the training data and training label within the training set simultaneously
        shuffle_plate = range(len(train_data))
        shuffle(shuffle_plate)
        new_train_data = []
        new_train_label = []

        for i in range(len(train_data)):
                index = shuffle_plate[i]
                new_train_data.append(train_data[index])
                new_train_label.append(train_label[index])

        # Assign back
        train_data = new_train_data
        train_label = new_train_label
    
        train_size = len(train_data)
        test_size = len(test_data)
	
	
        print "training set size: %u or %u" % (len(train_data),train_size)
        print(train_label)
        print "------------------------------------------------"
        print "test set size: %u or %u" % (len(test_data),test_size)
        print(test_label)
        print "------------------------------------------------"
        #print"actual train set length of rows: "
        #print(actual_train_input_len)
        #print"actual test set length of rows: "
        #print(actual_test_input_len)


if __name__ == '__main__':
        fileList = getMatFileList()
        getData(fileList)
	flatten_train_data = []
	flatten_train_label = []
	flatten_test_data = []
	flatten_test_label = []
	c = 0
	for row in train_data:
		flatten_train_data.extend(row)
		rowLen = len(row)
		flatten_train_label.extend([train_label[c]]*rowLen)
		c += 1
	c = 0
	for row in test_data:
		flatten_test_data.extend(row)
		rowLen = len(row)
		flatten_test_label.extend([test_label[c]]*rowLen)
		c += 1
	flatten_train_data = np.array(flatten_train_data, np.float64)
	flatten_train_label = np.array(flatten_train_label, np.float64)
	flatten_test_data = np.array(flatten_test_data, np.float64)
	flatten_test_label = np.array(flatten_test_label, np.float64)
	print(flatten_train_data.shape)
	print(flatten_train_label.shape)
	print(flatten_test_data.shape)
	print(flatten_test_label.shape)
	
	clf = SVC()
	clf.fit(flatten_train_data, flatten_train_label)
	SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    		 degree=3, gamma='auto', kernel='rbf',
    		max_iter=-1, probability=False, random_state=None, shrinking=True,
    		tol=0.001, verbose=False)

	#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    	#	decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    	#	max_iter=-1, probability=False, random_state=None, shrinking=True,
    	#	tol=0.001, verbose=False)
	
	prediction = clf.predict(flatten_test_data)
	#print(clf.predict(flatten_test_data[0:5]))
	acc_num = 0
	for i in range(len(flatten_test_label)):
		if flatten_test_label[i] == prediction[i]:
			acc_num += 1
	print "accuracy: %.6f" % ((float)(acc_num)/len(flatten_test_label))
