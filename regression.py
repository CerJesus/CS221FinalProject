"""
Beating the Bubble: Housing Prices in Ames, Iowa

Filename: regression.py
Authors:  Alexandre Bucquet, Jesus Cervantes, Alex Kim
Python 2.7

DESCRIPTION
This script defines a standard linear regression predictor and evaluates its
mean squared error.
"""

import math, random
from collections import defaultdict
import numpy  as np
import pandas as pd
from util import dotProduct, increment, evaluatePredictor, csvAsArray

# FUNCTIONS --------------------------------------------------------------------

### Global variables
col_names = []

# 
def error(example, weights, true):
    return (true - dotProduct(example, weights)) ** 2

def d_error(features, weights, trueVal):
	gradient = {}
	scale = 2*(dotProduct(features, weights) - trueVal)
	increment(gradient, scale, features)
	return gradient

# REGRESSION: Perform a linear regression 
def learnRegression(examples, numIters, stepSize):
	weights = defaultdict(int)

	for i in range(numIters):
		for x, y in examples:
			#print "x", len(x)
			features = featurize(x)

			gradient = d_error(features, weights, y)
			increment(weights, - stepSize, gradient)
		print 100.0*i/numIters
			#print weights

	def predictor(x):
		return dotProduct(featurize(x), weights)

	#print weights
	return predictor

# FEATURIZE
def featurize(x):
    features = defaultdict(int)
    for i in range(len(x)):
	if type(x[i]) == str:
	    #add an indicator feature
	    features[col_names[i] + x[i]] = 1

	elif math.isnan(x[i]):
	    features[col_names[i] + 'NA'] = 1

	else:
	    #add the number itself as our feature value
	    features[col_names[i]] = x[i]
	
    return features

# COMPUTATION ------------------------------------------------------------------

# Import the training and test data as numpy arrays
train_array = csvAsArray('train_updated.csv')
test_array  = csvAsArray('test.csv')

data_train = pd.read_csv('train_updated.csv')
col_names  = data_train.columns.tolist()[1:]
#print test_array.dtype.names

train_examples = []
for i in range(len(train_array)):
    input_size = range(len(train_array[i]) - 1)
    input_data = [train_array[i][j] for j in input_size]
    output     = train_array[i][len(train_array[0]) - 1]
    train_examples.append((input_data, output))


"""
train_examples = [ ( [train_array[i][j] for j in range(len(train_array[i]) - 1) ], train_array[i][len(train_array[0]) - 1]) for i in range(len(train_array))]
"""

# Evaluate the mean squared error of regression
regressionPredictor = learnRegression(train_examples, 10, 0.00000000001)
regression_error    = evaluatePredictor(regressionPredictor, train_examples)

# Print the results
print ""
print "----------"
print "REGRESSION"
print "----------"
print "Number of examples: ", len(train_examples)
print "Regression MSE:     ", regression_error
print ""
