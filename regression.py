#This is our logistic regression file
import math, random
from collections import defaultdict
import numpy
import pandas as pd


def dotProduct(list1, list2):

	if len(list1) != len(list2):
		raise Exception('dotProduct error: list length dont match')
	dot = 0
	for i in len(list1):
		dot += list1[i] * list2[i]
	return dot

def error(example, weights, true):
	return math.pow((true - dotProduct(example, weights)), 2)

def d_error(example, weights, true):
	print example
	scale = 2*(dotProduct(example, weights) - true)
	return [x * scale for x in example]

def learnRegression(examples, numIters, stepSize):
	weights = [0 for i in range(len(examples[0]))]


	for i in range(numIters):
		for x, y in examples:
			gradient = d_error(x, weights, y)
			weights -= [stepSize * element for element in gradient]

	def predictor(x):
		return dotProduct(x, tuple(weights))

	return predictor

def evaluatePredictor(predictor, examples):
    '''
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predictor| and returns the average error
    on the set of examples.
    '''
    error = 0.0
    for x, y in examples:
        error += math.pow((predictor(x) - y), 2)
    return error / len(examples)

## Processing training data
file_train = 'train.csv'
data_train = pd.read_csv(file_train)
#print(data_train.head())
train_array = data_train.as_matrix(columns=None)

train_examples = [ ( (train_array[i][j] for j in range(len(train_array[i]) - 1) ), train_array[i][80]/1000.0) for i in range(len(train_array))]
print train_array
'''for i in range(len(train_array)):
	cur_x = ()
	for j in range(len(train_array[i]) - 1):
		cur_x += (train_array[i][j], )
	train_examples += (cur_x,   train_array[i][80])'''

## Processing testing data
file_test = 'test.csv'
data_test = pd.read_csv(file_test)
#print(data.head())
test_array = data_test.as_matrix(columns=None)

#test_examples = ( ( (test_array[i][j] for j in range(len(test_array[i]) - 1) ), test_array[i][79]) for i in range(len(test_array)))

logisticPredictor = learnRegression(train_examples, 100, 0.1)
print "the training error for the regression is:", evaluatePredictor(logisticPredictor, train_examples)


