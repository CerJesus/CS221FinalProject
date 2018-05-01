#This is our baseline/oracle file
import math, random
from collections import defaultdict
import numpy
import pandas as pd


def learnBaseline(np_array):
	train_results = [np_array[i][80] for i in range(len(np_array))]
	#print(train_results)
	numpy.sort(train_results)
	median_test_data = train_results[len(train_results)/2 - 1]/1000.0
	print(median_test_data)

	def baselinePredictor(x):
		return median_test_data

	return baselinePredictor


def learnOracle(examples):
	examples_map = {}
	for x,y in examples:
		examples_map[x] = y

	def predictor(x):
		return examples_map[x]

	return predictor

def evaluatePredictor(predictor, examples):
    '''
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predict| and returns the average error
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
print train_array

train_examples = [ ( (train_array[i][j] for j in range(len(train_array[i]) - 1) ), train_array[i][80]/1000.0) for i in range(len(train_array))]
print len(train_examples)
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

baseline = learnBaseline(train_array)
oracle_train = learnOracle(train_examples)

print "baseline average train error:", evaluatePredictor(baseline, train_examples)
print "oracle average train error:", evaluatePredictor(oracle_train, train_examples)
