"""
Beating the Bubble: Housing Prices in Ames, Iowa

Filename: clustering.py
Authors:  Alexandre Bucquet, Jesus Cervantes, Alex Kim
Python 2.7

DESCRIPTION
This script performs k-means clustering on the dataset.
"""

import math, random
from collections import defaultdict
import numpy
import pandas as pd
import boostedtree
import regression
import kmeans
import util

## CONSTANTS
SGD_ITERS = 10
ETA = 0.00000000001
NUM_SPLITS = 10
NUM_CENTROIDS = 15
K_ITERS = 500

# COMPUTATION ------------------------------------------------------------------

def trainAndTest():

    # Import the training and test data as numpy arrays
    train_array = util.csvAsArray('train_updated.csv')
    test_array  = util.csvAsArray('test.csv')

    # Generate a list of (feature vector, value) tuples for the training data
    feature_names = util.getCsvHeaders('train_updated.csv')

    train_examples = []
    k_examples = []
    for i in range(len(train_array)):
        feature_count  = range(len(train_array[i]) - 1)
        feature_values = [train_array[i][j] for j in feature_count]
        feature_vector = util.featurize(feature_values, feature_names)
        output         = train_array[i][len(train_array[0]) - 1]
        train_examples.append((feature_vector, output))
        k_examples.append(feature_vector)

    # Cluster the data using k-means
    (centroids, assign, loss) = kmeans.kmeans(k_examples, NUM_CENTROIDS, K_ITERS)

    # Make clusters
    cluster_list = [ [] for _ in range(len(centroids))]

    for i in range(len(train_examples)):
    	cluster_list[assign[i]].append(train_examples[i])

    # Train a regression model on the training data (by cluster) 
    # and evaluate its mean squared error with the train data
    regression_error = 0
    predictor_list = []
    for cluster_points in cluster_list:
    	boostedRegressionPredictor = boostedtree.learnBoostedRegression(cluster_points, SGD_ITERS, ETA)
    	predictor_list.append(boostedRegressionPredictor)
    	regression_error += boostedtree.evaluatePredictor(boostedRegressionPredictor, cluster_points)*len(cluster_points)

    regression_error /= len(train_examples)

    # Print the results
    print ""
    print "------------------"
    print "CLUSTERED REGRESSION WITH BOOSTING"
    print "------------------"
    print "Number of examples: " + str(len(train_examples))
    print "Regression MSE:     " + str(regression_error)
    print ""

    return predictor_list, centroids, regression_error

trainAndTest()
