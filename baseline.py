# Beating the Bubble: Housing Prices in Ames, Iowa

# Filename: baseline.py
# Authors:  Alexandre Bucquet, Jesus Cervantes, Alex Kim
# Python 2.7

# DESCRIPTION
# This script evaluates the accuracy of a naive baseline predictor

import math, random
from collections import defaultdict
import numpy  as np
import pandas as pd

# FUNCTIONS --------------------------------------------------------------------

# BASELINE: Return the median sale price given an appropriate input tuple
def learnBaseline(np_array):
	sale_prices = [np_array[i][80] for i in range(len(np_array))]
	np.sort(sale_prices)
	median_sale_price = sale_prices[len(sale_prices) / 2 - 1] / 1000.0

	def baselinePredictor(input):
		return median_sale_price

	return baselinePredictor

# ORACLE: Return the true sale price given an appropriate input tuple
def learnOracle(examples):
	examples_map = {}
	for input, output in examples:
		examples_map[input] = output

	def predictor(x):
		return examples_map[x]

	return predictor

# PREDICTOR EVALUATION: Compute the mean squared error of a predictor function
# given a matrix of examples
def evaluatePredictor(predictor, examples):
	error = 0.0
	for input, output in examples:
		error += (predictor(input) - output) ** 2
	error /= len(examples)
	return error

# COMPUTATION ------------------------------------------------------------------

# Import the training data into a numpy matrix
file_train  = 'train.csv'
data_train  = pd.read_csv(file_train)
train_array = data_train.as_matrix(columns = None)

# Import the test data into a numpy matrix
file_test  = 'test.csv'
data_test  = pd.read_csv(file_test)
test_array = data_test.as_matrix(columns=None)

# Format the training data as a list of (input, output) tuples
train_examples = []
for i in range(len(train_array)):
	input_size = range(len(train_array[i]) - 1)
	input      = (train_array[i][j] for j in input_size)
	output	   = train_array[i][80] / 1000.0
	train_examples.append((input, output))

# Define predictor functions for baseline and oracle
baseline     = learnBaseline(train_array)
oracle_train = learnOracle(train_examples)

# Evaluate mean squared error of predictors
baseline_error = evaluatePredictor(baseline, train_examples)
oracle_error   = evaluatePredictor(oracle_train, train_examples)

# Print the results
print ""
print "--------"
print "BASELINE"
print "--------"
print "Number of examples:    ", len(train_examples)
print "Baseline (median) MSE: ", baseline_error
print "Oracle MSE:            ", oracle_error
print ""
