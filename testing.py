#This is our logistic regression file
import math, random
from collections import defaultdict
import numpy
import pandas as pd
import boostedtree
import regression
import kmeans


## CONSTANTS
SGD_ITERS = 10
ETA = 0.00000000001
NUM_SPLITS = 10


def r_squared(examples, predictor, model):
	prediction_error = model.evaluatePredictor(predictor, examples)*len(examples)

	outputs = []
	for i in range(len(examples)):
		outputs.append(examples[i][1])

	mean = 1.0*sum(outputs)/len(outputs)

	variance = 0
	for i in range(len(outputs)):
		variance += math.pow(outputs[i] - mean, 2)

	variance = 1.0*variance/len(outputs)

	return 1 - (prediction_error/variance)


def find_feature_values(examples):
	
	r_adj_list = []
	for i in range(len(examples[i][0])):
		#remove the feature
		modified_examples = examples[:]
		for j in range(len(modified_examples)):
			modified_examples[j][0].remove(i)

		#train a model
		boostedtree.learnRegression(modified_examples, SGD_ITERS, ETA)

		#compute the adjusted r^2
		n = len(examples)
		k = len(modified_examples[i][0])
		r_sq = r_squared(modified_examples, predictor, boostedtree)
		r_sq_adj = 1 - (1 - r_sq)((n-1)/(n - k - 1))

		#store it
		r_adj_list.append(r_sq_adj)

	return r_adj_list


## Processing training data
file_train = 'train_updated.csv'
data_train = pd.read_csv(file_train)
col_names = data_train.columns.tolist()[1:]

train_array = data_train.as_matrix(columns=None)

train_examples = [ ( [train_array[i][j] for j in range(len(train_array[i]) - 1) ], train_array[i][len(train_array[0]) - 1]) for i in range(len(train_array))]

## split the data, train and test
random.shuffle(train_examples)
for i in range(NUM_SPLITS):
	startTest = i*len(train_examples)/NUM_SPLITS
	endTest = (i+1)*len(train_examples)/NUM_SPLITS
	currentTrainExamples = train_examples[0:startTest] + train_examples[endTest:len(train_examples)]
	logisticPredictor = boostedtree.learnRegression(currentTrainExamples, SGD_ITERS, ETA)
	print "leaving out the", (i+1), "th segment of the data, the testing error for the regression is:", boostedtree.evaluatePredictor(logisticPredictor, train_examples[startTest:endTest])

#test_examples = ( ( (test_array[i][j] for j in range(len(test_array[i]) - 1) ), test_array[i][79]) for i in range(len(test_array)))

#logisticPredictor = learnRegression(train_examples, 10, 0.00000000001)
#print "the training error for the regression is:", evaluatePredictor(logisticPredictor, train_examples)
