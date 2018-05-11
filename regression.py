#This is our logistic regression file
import math, random
from collections import defaultdict
import numpy
import pandas as pd

### Global variables
col_names = []


### Helper functions
def dotProduct(vec1, vec2):
	dot = 0
	for k, v in vec1.iteritems():
		if k in vec2.keys():
			dot += v * vec2[k]
	return dot

def increment(vec1, scale, vec2):
	for f, v in vec2.items():
		if f in vec1.keys():
			vec1[f] += scale * vec2[f]
		else:
			vec1[f] = scale * vec2[f]


def error(example, weights, true):
	return math.pow((true - dotProduct(example, weights)), 2)

def d_error(features, weights, trueVal):
	#print example
	gradient = {}
	scale = 2*(dotProduct(features, weights) - trueVal)
	increment(gradient, scale, features)
	return gradient

def learnRegression(examples, numIters, stepSize):
	#print examples[0]
	weights = defaultdict(int)
	#print "weights", len(weights)

	for i in range(numIters):
		for x, y in examples:
			#print "x", len(x)
			features = featurize(x)

			gradient = d_error(features, weights, y)
			increment(weights, stepSize, gradient)

	def predictor(x):
		return dotProduct(featurize(x), weights)

	print weights
	return predictor

def featurize(x):
	features = defaultdict(int)

	for i in range(len(x)):
		if type(x[i]) == str:
			#add an indicator feature
			features[col_names[i] + x[i]] = 1
		else:
			#add the number itself as our feature value
			features[col_names[i]] = x[i]
	
	#print features
	return features

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
file_train = 'train_updated.csv'
data_train = pd.read_csv(file_train)
col_names = data_train.columns.tolist()[1:]

#print(data_train.head())
train_array = data_train.as_matrix(columns=None)

train_examples = [ ( [train_array[i][j] for j in range(len(train_array[i]) - 1) ], train_array[i][len(train_array[0]) - 1]) for i in range(len(train_array))]
#print train_examples
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

logisticPredictor = learnRegression(train_examples, 1, 0.1)
print "the training error for the regression is:", evaluatePredictor(logisticPredictor, train_examples)


