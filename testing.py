"""
Beating the Bubble: Housing Prices in Ames, Iowa

Filename: testing.py
Authors:  Alexandre Bucquet, Jesus Cervantes, Alex Kim
Python 2.7

DESCRIPTION
This script generates a validation set, trains a model, and evaluates its
error. Cross-validation is also enabled.
"""

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

# R SQUARED: Compute the r-squared value
def r_squared(examples, predictor):
	prediction_error = util.evaluatePredictor(predictor, examples)*len(examples)

	outputs = []
	for i in range(len(examples)):
		outputs.append(examples[i][1])

	mean = 1.0*sum(outputs)/len(outputs)

	variance = 0
	for i in range(len(outputs)):
		variance += math.pow(outputs[i] - mean, 2)

	variance = 1.0*variance/len(outputs)

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

def forward_selection():
	features_order = []
	best_r_list = []
	col_names_left = copy(col_names)
	p = len(featurized_examples[0][0])
	examples_list = [(defaultdict(int) , featurized_examples[j][1]) for j in range(p)]


	for k in range(p):
		# consider all p - k models that add a predictor to the very last model
		r_sq_list = []
		count = 1
		for var_name in col_names_left:
			for i in range(len(examples_list)):
				first_entry = examples_list[i][0]
				if var_name in featurized_examples[i][0]:
					first_entry[var_name] = featurized_examples[i][0][var_name]
				examples_list[i] = (first_entry, examples_list[i][1])

			predictor = boostedtree.learnBoostedRegression(examples_list, SGD_ITERS, ETA, NUM_TREES)

			#compute  r^2
			r_sq = r_squared(examples_list, predictor)

			#store it
			r_sq_list.append(r_sq)
			
			print "we have trained ", (count), " out of ", (len(col_names_left) - 1), "variables"
			count += 1

		# choose the best one and add it to the list
		max_val = -100
		max_ind = -1
		for i in range(len(col_names_left)):
			if r_sq_list[i] > max_val:
				max_val = r_sq_list[i]
				max_ind = i
		features_order.append(col_names_left[max_ind])
		col_names_left.pop(max_ind)
		best_r_list.append(max_val)

		print "We have considered ", (k+1), " out of ", (p-1), "variables"
	#select the best model out of the list of predictors
	print features_order
	print best_r_list

forward_selection()

## split the data, train and evaluation 
def testTrees():
	random.shuffle(featurized_examples)
	for i in range(NUM_SPLITS):
		startTest = i*len(featurized_examples)/NUM_SPLITS
		endTest = (i+1)*len(featurized_examples)/NUM_SPLITS
		currentTrainExamples = featurized_examples[0:startTest] + featurized_examples[endTest:len(featurized_examples)]
		logisticPredictor = boostedtree.learnRegression(currentTrainExamples, SGD_ITERS, ETA)
		print "leaving out the", (i+1), "the segment of the data, the validation error for the regression is:", boostedtree.evaluatePredictor(logisticPredictor, featurized_examples[startTest:endTest])

#test_examples = ( ( (test_array[i][j] for j in range(len(test_array[i]) - 1) ), test_array[i][79]) for i in range(len(test_array)))

#logisticPredictor = learnRegression(train_examples, 10, 0.00000000001)
#print "the training error for the regression is:", evaluatePredictor(logisticPredictor, train_examples)
