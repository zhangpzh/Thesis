import scipy.io as sio
import os
import numpy as np
import tensorflow as tf
from random import shuffle
from math import ceil
#from tensorflow.models.rnn import rnn, rnn_cell
from peizhen_rnn import rnn, rnn_cell
import sys
import string

# File directory
matFileRootDirect = u'/home/peizhen/HON4D_feature_part_10_frame_max'
accuracyRecordTxtPth = u'/home/peizhen/expr_dir/experiment_results.txt'
distributionTxtDirectPth = u'/home/peizhen/expr_dir/dataset_distribution'
confusionMatrixRecordTxtPth = u'/home/peizhen/expr_dir/confusion_matrice.txt'

# Specify which distribution solution to use (1/2/3/4)
distributionTxtNum = 0


# size of training set and test set
train_size = 0
test_size = 0
	
# train_size x n_input
train_data = []
# test_size x n_input
test_data = []

# train_size x class_num
train_label = []
# test_size x class_num
test_label = []

# Record the effective length of each training data or test data that will be thrown to rnn
actual_train_input_len = []
actual_test_input_len = []

# Parameters
training_iters = 200000 	# Iteration number
batch_size = 20 		# (will be assigned while initialization)
basic_learning_rate = 0.1	# Basic learning rate

display_step = 10

# Network parameters
n_input = 3240
n_steps = 34  		# at most 25 + 9 = 34, at least 10

#n_hidden = 2048 	# hidden variables' number. (will be assigned while initialization)

n_classes = 7  		# peizhen's kinect dataset total classes (class index: 0~6)
#coef_rnn = 1		# coefficient of rnn: 1 for GRU, 2 for LSTM (will be assigned  while initialization)


# Get training ".mat" file name and test ".mat" file name 
def getMatFileList():
	global distributionTxtDirectPth
	distributionTxtPth = os.path.join(distributionTxtDirectPth,'distribution_0%s.txt' % (sys.argv[4]))
		
	
	training_matFileList = []
	test_matFileList = []
	all_matFileList = []
	for i in range(6):
		all_matFileList.append(i+1)
	for i in range(81):
		all_matFileList.append(i+15)
	all_matFileList = ['video%02u_HON4D_feature.mat' % (ele) for ele in all_matFileList]
	
	# Read test video number to list
	handle = open(distributionTxtPth,'r')
	test_matFileList = handle.readline().replace('\n','').split(' ') 
	test_matFileList = ['video%02u_HON4D_feature.mat' % (int(ele)) for ele in test_matFileList]
	
	
	# Get training video number to list
	training_matFileList = list(set(all_matFileList).difference(set(test_matFileList)))
	
	return [training_matFileList, test_matFileList]


# Construct training set and test set
def getData(matFileList):
	# Global declaration for function to change the value of following global variables
	global train_data
	global train_size
	global train_label
	global test_data
	global test_size
	global test_label
	global actual_train_input_len
	global actual_test_input_len

	
	training_matFileList = matFileList[0]
	test_matFileList = matFileList[1]
	
	# Get train_data
	for i in range(len(training_matFileList)):
		matFilePth = os.path.join(matFileRootDirect,training_matFileList[i])
		data = sio.loadmat(matFilePth)
		curVideoData = data['current_video_data']
		features = curVideoData[0][0][0]	# Get all features of current video -> k x 3240
		label = curVideoData[0][0][1][0][0]	# Get label of current video	    -> an integer

		# Transfer current video information to several rows of vectors(each vector with at most 25 elements)
		# Each element is a 1 x 3240 vector
		
		# Used for counting how many rows in train_data that are taken by current video
		n_takenRows = 0	

		thisRowFeature = []
		origIndex = 0
		while origIndex < len(features):
			thisRowFeature.append(features[origIndex])
			if len(thisRowFeature) == 25:
				train_data.append(thisRowFeature)
				thisRowFeature = []
				n_takenRows += 1
			origIndex += 4

		# If the length of last row is less than 25
		if len(thisRowFeature) > 0:
			# If the length is longer than 10, append it to "train_data" as a new row
			if len(thisRowFeature) >= 10:
				n_takenRows += 1
				train_data.append(thisRowFeature)
			# ... shorter than 10, extend it to the last row of "train_data"
			else:
				lastIndex = len(train_data)-1
				train_data[lastIndex].extend(thisRowFeature)
			thisRowFeature = []
		
		empty_label = [0]*n_classes
		empty_label[label] = 1

		# Append labels
		train_label.extend([empty_label]*n_takenRows)
		
		
	# Get test_data
	for i in range(len(test_matFileList)):
		matFilePth = os.path.join(matFileRootDirect,test_matFileList[i])
		data = sio.loadmat(matFilePth)
		curVideoData = data['current_video_data']
		features = curVideoData[0][0][0]	# Get all features of current video -> k x 3240
		label = curVideoData[0][0][1][0][0]	# Get label of current video	    -> an integer

		# Transfer current video information to several rows of vectors(each vector with at most 25 elements)
		# Each element is a 1 x 3240 vector
		
		# Used for counting how many rows in test_data that are taken by current video
		n_takenRows = 0	

		thisRowFeature = []
		origIndex = 0
		while origIndex < len(features):
			thisRowFeature.append(features[origIndex])
			if len(thisRowFeature) == 25:
				test_data.append(thisRowFeature)
				thisRowFeature = []
				n_takenRows += 1
			origIndex += 4

		# If the length of last row is less than 25
		if len(thisRowFeature) > 0:
			# If the length is longer than 10, append it to "test_data" as a new row
			if len(thisRowFeature) >= 10:
				n_takenRows += 1
				test_data.append(thisRowFeature)
			# ... shorter than 10, extend it to the last row of "test_data"
			else:
				lastIndex = len(test_data)-1
				test_data[lastIndex].extend(thisRowFeature)
			thisRowFeature = []
		
		empty_label = [0]*n_classes
		empty_label[label] = 1

		# Append labels
		test_label.extend([empty_label]*n_takenRows)

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

	
	# Fix every row of train_data and test_data to length of 34 by concatenating certain numbers of zeros vector with shape -> 1x3240
	# Of course, original length of each row will be recorded before such "fix" operation is done

	actual_train_input_len = [len(train_data[i]) for i in range(len(train_data))]
	actual_test_input_len = [len(test_data[i]) for i in range(len(test_data))]

	zeroHON4DFeature = [0]*3240
	
	# Fix training set
	for i in range(len(train_data)):
		n_padding = n_steps-actual_train_input_len[i]
		while n_padding > 0:
			n_padding -= 1
			train_data[i].append(zeroHON4DFeature)
	# Fix test set
	for i in range(len(test_data)):
		n_padding = n_steps-actual_test_input_len[i]
		while n_padding > 0:
			n_padding -= 1
			test_data[i].append(zeroHON4DFeature)

	# Transfer train_data, test_data, train_label, test_label, actual_train_input_len, actual_test_input_len into ndarray in numpy
	train_data = np.array(train_data,np.float64)
	test_data = np.array(test_data,np.float64)
	train_label = np.array(train_label,np.float64)
	test_label = np.array(test_label,np.float64)
	actual_test_input_len = np.array(actual_test_input_len, np.float64)
	actual_train_input_len = np.array(actual_train_input_len, np.float64)
	
	#print train_data.shape
	#print test_data.shape
	
	print "training set size: %u or %u" % (train_data.shape[0],train_size)
	print(train_label)
	print "------------------------------------------------"
	print "test set size: %u or %u" % (test_data.shape[0],test_size)
	print(test_label)
	print "------------------------------------------------"
	print"actual train set length of rows: "
	print(actual_train_input_len)
	print"actual test set length of rows: "
	print(actual_test_input_len)

	
	#test_lable_out_list = train_label.argmax(1).tolist()
	#activityPrediction = [0]*7
	#for i in range(7):
	#	activityPrediction[i] = test_lable_out_list.count(i)
	#print "current batch predicting labels count(0~7): ", activityPrediction
	
	




if __name__ == '__main__':

	# Global declaration
	global train_data
	global train_size
	global train_label
	global test_data
	global test_size
	global test_label
	global actual_train_input_len
	global actual_test_input_len

	global training_iters
	global batch_size
	global display_step
	global n_input
	global n_steps
	global n_classes
	global basic_learning_rate
	
	fileList = getMatFileList()
        getData(fileList)
	
	n_hidden = int(sys.argv[1])
	basic_learning_rate = string.atof(sys.argv[2])
	coef_rnn = int(sys.argv[3])
	
	# tf Graph input
	x = tf.placeholder("float", [None, n_steps, n_input])
	# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
	istate = tf.placeholder("float", [None, coef_rnn*n_hidden])
	
	y = tf.placeholder("float", [None, n_classes])
	
	# Define weights
	weights = {
		'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
	    	'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
	}
	biases = {
	    	'hidden': tf.Variable(tf.random_normal([n_hidden])),
	    	'out': tf.Variable(tf.random_normal([n_classes]))
	}
	
	# what timesteps we want to stop at, notice it's different for each batch hence dimension of [batch]
	early_stop = tf.placeholder(tf.int32, [None])
	keep_prob = tf.placeholder('float')
	
	def RNN(_X, _istate, _weights, _biases, output_keep_prob):
		# input shape: (batch_size, n_steps, n_input)
	    	_X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
	    	# Reshape to prepare input to hidden activation
	    	_X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
	    	# Linear activation
	    	_X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']
	
		if coef_rnn == 2:
	    		lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
		else:
	    		lstm_cell = rnn_cell.GRUCell(n_hidden)	
		

		lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob)
	
	    	# Split data because rnn cell needs a list of inputs for the RNN inner loop
	    	_X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)
	
	    	# Get lstm cell output
	    	outputs, states = rnn(lstm_cell, _X, initial_state=_istate, sequence_length=early_stop)
	
	    	# Linear activation
	    	# Get inner loop last output
	    	return tf.matmul(outputs[-1], _weights['out']) + _biases['out']
	
	batch = tf.Variable(0, trainable=False)
	
	# Decay once per epoch, using an exponential schedule starting at 0.1
	learning_rate = tf.train.exponential_decay(
		basic_learning_rate,
		batch * batch_size, 
		train_size, 
		0.95,
		staircase=True)
	
	pred = RNN(x, istate, weights, biases, keep_prob)
	
	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=batch) # Adam Optimizer
	
	# Evaluate model
	preLabel = tf.argmax(pred,1)
	correct_pred = tf.equal(preLabel, tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
	# Initializing the variables
	init = tf.initialize_all_variables()

	formatStr = '%u\t%.3f\t%s\t%.6f\t%s\n'

	# Conduct rnn
	with tf.Session() as sess:
		sess.run(init)
		step = 1
		test_lable_out = sess.run(preLabel, feed_dict={early_stop: actual_test_input_len, x: test_data, y: test_label,
                                                            istate: np.zeros((len(test_data), coef_rnn*n_hidden)), keep_prob:1.0})
		test_lable_out_list = test_lable_out.tolist()
		activityPrediction = [0]*7
		for i in range(7):
			activityPrediction[i] = test_lable_out_list.count(i)
		print "current batch predicting labels count(0~7): ", activityPrediction
		print "Testing Accuracy:", sess.run(accuracy, feed_dict={early_stop: actual_test_input_len, x: test_data, y: test_label,
                                                            istate: np.zeros((len(test_data), coef_rnn*n_hidden)), keep_prob:1.0})
		previous_offset = 0

		# Keep training
		while step * batch_size	< training_iters:
			offset = (step * batch_size)%(train_size-batch_size)
			#The network has go through the training data batches once. Then shuffle the train_data and train_label simultaneously
			if offset < previous_offset:
				platte = np.arange(train_size)
				np.random.shuffle(platte)
				train_data = np.take(train_data, platte, axis=0)
				train_label = np.take(train_label, platte, axis=0)
				actual_train_input_len = np.take(actual_train_input_len, platte, axis=0) 
				
			previous_offset = offset
			batch_xs = train_data[offset:(offset+batch_size),...]
			batch_ys = train_label[offset:(offset+batch_size)]

			# Get actual length of batch datas
			e_stop = actual_train_input_len[offset:(offset+batch_size)]

			sess.run(optimizer, feed_dict={early_stop: e_stop, x: batch_xs, y: batch_ys,
						istate: np.zeros((batch_size, coef_rnn*n_hidden)), keep_prob:0.5})
			if step % display_step == 0:
            			# Calculate batch accuracy
            			acc = sess.run(accuracy, feed_dict={early_stop: e_stop, x: batch_xs, y: batch_ys,
            			                                    istate: np.zeros((batch_size, coef_rnn*n_hidden)), keep_prob:1.0})
            			# Calculate batch loss
            			loss = sess.run(cost, feed_dict={early_stop: e_stop, x: batch_xs, y: batch_ys,
            			                                 istate: np.zeros((batch_size, coef_rnn*n_hidden)), keep_prob:1.0})
            			print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
            			      ", Training Accuracy= " + "{:.5f}".format(acc)
				
			#if step % 300 == 0:
			#	test_lable_out = sess.run(preLabel, feed_dict={early_stop: actual_test_input_len, x: test_data, y: test_label,
                        #                                    istate: np.zeros((len(test_data), coef_rnn*n_hidden)), keep_prob:1.0})
			#	test_lable_out_list = test_lable_out.tolist()
			#	activityPrediction = [0]*7
			#	for i in range(7):
			#		activityPrediction[i] = test_lable_out_list.count(i)
			#	print "current batch predicting labels count(0~7): ", activityPrediction

			#	print "Testing Accuracy:", sess.run(accuracy, feed_dict={early_stop: actual_test_input_len, x: test_data, y: test_label,
                        #                                    istate: np.zeros((len(test_data), coef_rnn*n_hidden)), keep_prob:1.0})

			step += 1

			# Write into txt file
			if step*batch_size  >= training_iters:
				# Record accuracy
				[acc,prediction,actually] = sess.run([accuracy,preLabel,tf.argmax(y,1)], feed_dict={early_stop: actual_test_input_len, x: test_data, y: test_label,
                                                            istate: np.zeros((len(test_data), coef_rnn*n_hidden)), keep_prob:1.0})

				output = open(accuracyRecordTxtPth, 'a')
				rnn_name = 'GRU'
				if coef_rnn == 2:
					rnn_name = 'LSTM'
				info = formatStr % (n_hidden,basic_learning_rate,rnn_name,acc,sys.argv[4])
				output.write(info)
				output.close()

				# Record confusion matrix
				confusion_matrix =  np.zeros((7,7))
				# Count the occurrence number of each label in y	
				row_sum = np.zeros(7)

				for i in range(len(prediction)):
					actualLabelOfCurIndex = int(actually[i])
					predictedLabelOfCurIndex = int(prediction[i])
					confusion_matrix[actualLabelOfCurIndex][predictedLabelOfCurIndex] += 1
					row_sum[actualLabelOfCurIndex] += 1
				for i in range(7):
					for j in range(7):
						confusion_matrix[i][j] /= row_sum[i]

				output = open(confusionMatrixRecordTxtPth,'a')
				output.write('\n---------------------\n')
				output.write(info)
				for i in range(7):
					for j in range(7):
						output.write("%.2f\t" % (confusion_matrix[i][j]))
					output.write('\n')
				output.close()
				
