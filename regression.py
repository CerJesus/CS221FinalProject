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
from util import dotProduct, increment, lossGradient, featurize, \
        evaluatePredictor, csvAsArray, getCsvHeaders

# LEARNING FUNCTIONS -----------------------------------------------------------

# LOSS GRADIENT: Return the gradient of the training loss with respect to the
# weight vector for a given example (features, true_value)
def lossGradient(features, weights, true_value):
    gradient = {}
    scale = 2 * (dotProduct(features, weights) - true_value)
    increment(gradient, scale, features)
    return gradient

def lassoLossGradient(features, weights, true_value, tuning_parameter): #, tuning_parameter):

    # TODO: temp hard-coded parameter
    #tuning_parameter = .5

    # Standard squared loss
    gradient = {}
    scale =  2 * (dotProduct(features, weights) - true_value)

    # Lasso term: add gradient of the lasso term to the scaling factor (i.e.
    # add gradient of |tuning_parameter| * (1-norm of weights)
    weight_signs = [np.sign(weights[w]) for w in weights]

    for w in weights:
        gradient[w] = tuning_parameter * np.sign(weights[w])

    increment(gradient, scale, features)
    return gradient

def regularizationLossGradient(features, weights, true_value, tuning_parameter): #, tuning_parameter):

    # TODO: temp hard-coded parameter
    #tuning_parameter = .5

    # Standard squared loss
    gradient = {}
    scale =  2 * (dotProduct(features, weights) - true_value)

    # Regu;arization term: add gradient of the regularization term to the scaling factor (i.e.
    # add gradient of |tuning_parameter| * (2-norm of weights)^2 
    increment(gradient, tuning_parameter, weights)

    increment(gradient, scale, features)
    return gradient

# REGRESSION: Learn a linear regression model and return the predicted sale
# price given an input tuple
def learnRegression(examples, numIters, stepSize, tuning_parameter):
    weights = defaultdict(int)

    print ""
    for i in range(numIters):
        for x, y in examples:
            gradient = regularizationLossGradient(x, weights, y, tuning_parameter)
            increment(weights, -stepSize, gradient)
        print "Training progress: " + str(100.0 * (i + 1) / numIters) + "%"

    def predictor(x):
        return dotProduct(x, weights)

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

    random.shuffle(train_examples)
    test = train_examples[:len(train_examples)/10]
    train = train_examples[len(train_examples)/10:]

    # Train a regression model on the training data and evaluate its mean
    # squared error with the test data
    for tuning_parameter in range(5, 21, 5):
        tuning_parameter = 1.0 * tuning_parameter/10
        regressionPredictor = learnRegression(train, 500, 0.00000000001, tuning_parameter)
        regression_error    = evaluatePredictor(regressionPredictor, test)

        # Print the results
        print ""
        print "----------"
        print "REGRESSION"
        print "----------"
        print "Lambda (lasso): ", tuning_parameter
        print "Number of examples: ", len(train_examples)
        print "Regression MSE:     ", regression_error
        print ""

#trainAndEvaluate()
