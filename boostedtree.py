"""
Beating the Bubble: Housing Prices in Ames, Iowa

Filename: boostedtree.py
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
from util import dotProduct, increment, featurize, evaluatePredictor, \
        csvAsArray, getCsvHeaders

# LEARNING FUNCTIONS -----------------------------------------------------------

# LOSS GRADIENT: Return the gradient of the training loss with respect to the
# weight vector for a given example (features, true_value)
def lossGradient(features, weights, true_value):
    gradient = {}
    scale = 2 * (dotProduct(features, weights) - true_value)
    increment(gradient, scale, features)
    return gradient

# BOOSTED REGRESSION: Learn a linear regression model using boosted trees and
# return the predicted sale price given an input tuple
def learnBoostedRegression(examples, num_iters, step_size, num_trees):
    list_weights = [defaultdict(int) for _ in range(num_trees)]
    objectives = [cur[1] for cur in examples]

    for k in range(num_trees):
        print ""
        print "TREE " + str(k + 1) + " OF " + str(num_trees)
        curWeights = defaultdict(int)
        for i in range(num_iters):
            for ind in range(len(examples)):
                x = examples[ind][0]
                gradient = lossGradient(x, curWeights, objectives[ind])
                increment(curWeights, - step_size, gradient)
            print "Training progress: " + str(100.0 * (i + 1) /num_iters) + "%"
    	
        list_weights[k] = curWeights

        for j in range(len(examples)):
            x, y = examples[j]
            objectives[j] = objectives[j] - dotProduct(x, curWeights)
        	
        print "COMPLETE"

    def predictor(x):
        return sum(dotProduct(x, curWeight) for curWeight in list_weights)

    return predictor

# COMPUTATION ------------------------------------------------------------------

def trainAndEvaluate():

    # Import the training data as a numpy array
    train_array = csvAsArray('train_updated.csv')

    # Generate a list of (feature vector, value) tuples for the training data
    feature_names = getCsvHeaders('train_updated.csv')
    train_examples = []
    for i in range(len(train_array)):
        feature_count  = range(len(train_array[i]) - 1)
        feature_values = [train_array[i][j] for j in feature_count]
        feature_vector = featurize(feature_values, feature_names)
        output         = train_array[i][len(train_array[0]) - 1]
        train_examples.append((feature_vector, output))

    # Train a regression model on the training data and evaluate its mean
    # squared error with the test data
    boostedRegressionPredictor = learnBoostedRegression(train_examples, 10, \
            0.00000000001, num_trees=5)
    regression_error = evaluatePredictor(boostedRegressionPredictor, \
            train_examples)

    # Print the results
    print ""
    print "------------------"
    print "BOOSTED REGRESSION"
    print "------------------"
    print "Number of examples: " + str(len(train_examples))
    print "Regression MSE:     " + str(regression_error)
    print ""

trainAndEvaluate()
