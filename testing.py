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
from copy import deepcopy
import cPickle as pickle
#import regression
#import kmeans

## CONSTANTS
SGD_ITERS = 500
ETA = 0.0000001
NUM_SPLITS = 10
NUM_TREES = 5

# R SQUARED: Compute the r-squared value
def r_squared(examples, predictor):
	prediction_error = util.evaluatePredictor(predictor, examples)

	outputs = []
	for i in range(len(examples)):
		outputs.append(examples[i][1])

	mean = 1.0*sum(outputs)/len(outputs)

	def mean_pred(x):
		return mean

	variance = util.evaluatePredictor(mean_pred, examples)

	'''variance = 0
	for i in range(len(outputs)):
		variance += math.pow(outputs[i] - mean, 2)

	variance = 1.0*variance/len(outputs)'''

	return 1 - (prediction_error/variance)


def find_feature_values(examples, col_names):
	
	r_adj_list = []

	random.shuffle(examples)
	n = len(examples)
	print len(examples[0][0])

	test = examples[:n/10]
	train = examples[n/10:]

	for i in range(len(col_names)):
		#remove the feature
		modified_examples = deepcopy(train)
		for j in range(len(modified_examples)):
			#modified_examples[j][0].remove(i)
			for feature_name, value in modified_examples[j][0].iteritems():
				if col_names[i] in feature_name: 
					del modified_examples[j][0][feature_name]
					break
		
		print len(train[0][0]), len(modified_examples[0][0])
		#train a model
		predictor = boostedtree.learnBoostedRegression(modified_examples, SGD_ITERS, ETA/(2*len(col_names)), NUM_TREES)\

		print modified_examples[0]

		#compute the adjusted r^2
		k = len(modified_examples[i][0])
		r_sq = r_squared(test, predictor)
		r_sq_adj = 1 - (1 - r_sq)*((n-1)/(n - k - 1))
		print r_sq_adj

		#store it
		r_adj_list.append(r_sq_adj)
		print "Removed variable ", (i+1), " out of ", (len(col_names) - 1)

	return r_adj_list

## Processing training data
file_train = 'train_updated.csv'
data_train = pd.read_csv(file_train)
col_names = data_train.columns.tolist()
col_names = col_names[:len(col_names) - 1]

train_array = data_train.as_matrix(columns=None)
print train_array

train_examples = [ ( [train_array[i][j] for j in range(len(train_array[i]) - 1) ], train_array[i][len(train_array[0]) - 1]) for i in range(len(train_array))]

featurized_examples = [ (util.featurize(train_examples[i][0], col_names), train_examples[i][1]) for i in range(len(train_examples))]
#print featurized_examples

## evaluate the value of different features
def orderFeatures():
	n = len(col_names)
	train_copy = deepcopy(featurized_examples)
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
	col_names_left = deepcopy(col_names)
	p = len(featurized_examples[0][0])
	current_list = [(defaultdict(int) , featurized_examples[j][1]) for j in range(p)]


	for k in range(p):
		# consider all p - k models that add a predictor to the very last model
		r_sq_list = []
		count = 1
		for var_name in col_names_left:
			examples_list = [(defaultdict(int) , featurized_examples[j][1]) for j in range(p)]
			for i in range(len(examples_list)):
				first_entry = examples_list[i][0]
				for feature_name, value in featurized_examples[i][0].iteritems():
					if var_name in feature_name:
						first_entry[feature_name] = featurized_examples[i][0][feature_name]
				util.increment(first_entry, 1, current_list[i][0])
				examples_list[i] = (first_entry, examples_list[i][1])

			print " "
			print examples_list[0:5]
			predictor = boostedtree.learnBoostedRegression(examples_list, SGD_ITERS, ETA, NUM_TREES)

			#compute r^2
			r_sq = r_squared(examples_list, predictor)
			print r_sq

			#store it
			r_sq_list.append(r_sq)
			
			print "we have trained ", (count), " out of ", (len(col_names_left)), "variables"
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
		
		# adding the features to the current list
		for i in range(len(current_list)):
			first_entry = current_list[i][0]
			for feature_name, value in featurized_examples[i][0].iteritems():
				if features_order[len(features_order) - 1] in feature_name:
					first_entry[feature_name] = featurized_examples[i][0][feature_name]
				current_list[i] = (first_entry, current_list[i][1])


		print "We have considered ", (k+1), " out of ", (p-1), "variables"
		print current_list[0]
	#select the best model out of the list of predictors
	print features_order
	print best_r_list
	pickle.dump((features_order, best_r_list), open("sorted_features.p","wb"))
	#(features_order, best_r_list) = pickle.load(open("sorted_features.p","rb"))

## split the data, train and evaluation 
def testTrees():
	random.shuffle(featurized_examples)
	for j in range(NUM_TREES):
		for i in range(0, NUM_SPLITS, 2):
			startTest = i*len(featurized_examples)/NUM_SPLITS
			endTest = (i+1)*len(featurized_examples)/NUM_SPLITS
			currentTrainExamples = featurized_examples[0:startTest] + featurized_examples[endTest:len(featurized_examples)]
			logisticPredictor = boostedtree.learnBoostedRegression(currentTrainExamples, SGD_ITERS, ETA, 5)
			print "leaving out the", (i+1), "the segment of the data, the validation error for the regression is:", boostedtree.evaluatePredictor(logisticPredictor, featurized_examples[startTest:endTest])

testTrees()

#test_examples = ( ( (test_array[i][j] for j in range(len(test_array[i]) - 1) ), test_array[i][79]) for i in range(len(test_array)))

#logisticPredictor = learnRegression(train_examples, 10, 0.00000000001)
#print "the training error for the regression is:", evaluatePredictor(logisticPredictor, train_examples)
