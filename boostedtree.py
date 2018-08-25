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
import os.path
from collections import defaultdict
import numpy  as np
import pandas as pd
import cPickle as pickle
import regression
from util import dotProduct, increment, lossGradient, featurize, \
        evaluatePredictor, csvAsArray, getCsvHeaders

# LEARNING FUNCTIONS -----------------------------------------------------------

VERBOSE = True
SAVE = False
cross_val_seg = 1

# BOOSTED REGRESSION: Learn a linear regression model using boosted trees and
# return the predicted sale price given an input tuple
def learnBoostedRegression(examples, num_iters, step_size, num_trees):
    list_weights = []
    objectives = [cur[1] for cur in examples]

    filename = "boostedtree_" + str(num_trees - 1) + "_" + str(cross_val_seg) + ".p"
    if num_trees > 1 and SAVE:
        (list_weights, num_trees_prev, num_iters_prev) = pickle.load(open(os.path.join("boostedtree_weights", filename), "rb"))

    for k in range(num_trees):
        if k >= len(list_weights):
            print ""
            print "TREE " + str(k + 1) + " OF " + str(num_trees)
            curWeights = defaultdict(int)
            for i in range(num_iters):
                for ind in range(len(examples)):
                    x = examples[ind][0]
                    gradient = regression.lassoLossGradient(x, curWeights, objectives[ind], .5)
                    increment(curWeights, - step_size/(i+1), gradient)
                if VERBOSE: print "Training progress: " + str(100.0 * (i + 1) /num_iters) + "%"

            list_weights.append(curWeights)
        else:
            curWeights = list_weights[k]
        
        for j in range(len(examples)):
            x, y = examples[j]
            objectives[j] = objectives[j] - dotProduct(x, curWeights)

        if VERBOSE: print "COMPLETE"

    if SAVE: 
        filename = "boostedtree_" + str(num_trees) + "_" + str(cross_val_seg) + ".p"
        pickle.dump((list_weights, num_trees, num_iters), open(os.path.join("boostedtree_weights", filename), "wb"))

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

    random.shuffle(train_examples)
    test = train_examples[:len(train_examples)/10]
    train_examples = train_examples[len(train_examples)/10:]

    # Train a regression model on the training data and evaluate its mean
    # squared error with the test data
    boostedRegressionPredictor = learnBoostedRegression(train_examples, 500, \
            0.000000001, num_trees=5)
    regression_error = evaluatePredictor(boostedRegressionPredictor, \
            test)

    # Print the results
    print ""
    print "------------------"
    print "BOOSTED REGRESSION"
    print "------------------"
    print "Number of examples: " + str(len(train_examples))
    print "Regression MSE:     " + str(regression_error)
    print ""

#trainAndEvaluate()
