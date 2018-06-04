#This is our logistic regression file
import math, random
from collections import defaultdict
import numpy
import pandas as pd
import boostedtree
import util
from copy import copy
#import regression
#import kmeans


## CONSTANTS
SGD_ITERS = 10
ETA = 0.00000000001
NUM_SPLITS = 10
NUM_TREES = 2


def r_squared(examples, predictor):
	prediction_error = util.evaluatePredictor(predictor, examples) * len(examples)

	outputs = []
	for i in range(len(examples)):
		outputs.append(examples[i][1])

	mean = 1.0*sum(outputs)/len(outputs)

	variance = 0
	for i in range(len(outputs)):
		variance += math.pow(outputs[i] - mean, 2)

	variance = 1.0*variance

	print prediction_error/variance
	return 1 - (prediction_error/variance)


def find_feature_values(examples, col_names):
	
	r_adj_list = []
	for i in range(len(col_names)):
		#remove the feature
		modified_examples = copy(examples)
		for j in range(len(modified_examples)):
			#modified_examples[j][0].remove(i)
			cur = modified_examples[j][0]
			if col_names[i] in cur: cur.pop(col_names[i])

		#train a model
		predictor = boostedtree.learnBoostedRegression(modified_examples, SGD_ITERS, ETA, NUM_TREES)

		#compute the adjusted r^2
		n = len(examples)
		k = len(modified_examples[i][0])
		r_sq = r_squared(modified_examples, predictor)
		r_sq_adj = 1 - (1 - r_sq)*((n-1)/(n - k - 1))

		#store it
		r_adj_list.append(r_sq_adj)
		print "Removed variable ", (i+1), " out of ", (len(col_names) - 1)

	return r_adj_list


## Processing training data
file_train = 'train_updated.csv'
data_train = pd.read_csv(file_train)
col_names = data_train.columns.tolist()[1:]

train_array = data_train.as_matrix(columns=None)

train_examples = [ ( [train_array[i][j] for j in range(len(train_array[i]) - 1) ], train_array[i][len(train_array[0]) - 1]) for i in range(len(train_array))]

featurized_examples = [ (util.featurize(train_examples[i][0], col_names), train_examples[i][1]) for i in range(len(train_examples))]

## evaluate the value of different features
def orderFeatures():
	n = len(col_names)
	train_copy = copy(featurized_examples)
	col_names_cpy = col_names[:]
	r_sq = []
	order = []

	for i in range(n):
		r_adj_list = find_feature_values(train_copy, col_names_cpy)
		worst = -1
		min_val = 100
		for j in range(len(r_adj_list)):
			if min_val > r_adj_list[j]:
				worst = j
				min_val = r_adj_list[j]

		worst_feature = col_names_cpy[worst]
		col_names_cpy.pop(worst)
		order.append(worst)

		for j in range(len(train_copy)):
				#train_copy[j][0].remove(worst_feature)
				cur = train_copy[j][0]
				if worst_feature in cur:
					cur.pop(worst_feature)

		print r_adj_list

	print order

orderFeatures()
#test_examples = ( ( (test_array[i][j] for j in range(len(test_array[i]) - 1) ), test_array[i][79]) for i in range(len(test_array)))

#logisticPredictor = learnRegression(train_examples, 10, 0.00000000001)
#print "the training error for the regression is:", evaluatePredictor(logisticPredictor, train_examples)
