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
SGD_ITERS = 500
ETA = 0.00000000001
NUM_SPLITS = 10
NUM_CENTROIDS = 10
K_ITERS = 500

# COMPUTATION ------------------------------------------------------------------

def trainAndTest():

    # Import the training and test data as numpy arrays
    train_array = util.csvAsArray('data/train_updated.csv')
    test_array  = util.csvAsArray('data/test.csv')

    # Generate a list of (feature vector, value) tuples for the training data
    feature_names = util.getCsvHeaders('data/train_updated.csv')

    train_examples = []
    k_examples = []
    

    for i in range(len(train_array)):
        feature_count  = range(len(train_array[i]) - 1)
        feature_values = [train_array[i][j] for j in feature_count]
        feature_vector = util.featurize(feature_values, feature_names)
        output         = train_array[i][len(train_array[0]) - 1]
        train_examples.append((feature_vector, output))

    random.shuffle(train_examples)

    for i in range(1, NUM_SPLITS, 2):
        startTest = i*len(train_examples)/NUM_SPLITS
        endTest = (i+1)*len(train_examples)/NUM_SPLITS
        currentTrain = train_examples[0:startTest] + train_examples[endTest:len(train_examples)]
        currentTest = train_examples[startTest:endTest]

        # Cluster the data using k-means
        (centroids, assign, loss, loss_list, centroid_vals) = kmeans.kmeans(currentTrain, NUM_CENTROIDS, K_ITERS)

        # Make clusters
        cluster_list = [ [] for _ in range(len(centroids))]

        for j in range(len(currentTrain)):
        	cluster_list[assign[i]].append(currentTrain[j])

        # Train a regression model on the training data (by cluster) 
        # and evaluate its mean squared error with the train data
        regression_error = 0
        predictor_list = []
        pre_computed_centroid_dots = [util.dotProduct(centroids[k], centroids[k]) for k in range(len(centroids))]

        for cluster_points in cluster_list:
        	boostedRegressionPredictor = boostedtree.learnBoostedRegression(cluster_points, SGD_ITERS, ETA, 5, 0)
        	predictor_list.append(boostedRegressionPredictor)
        	
    	def predictor(x):
    		centroid_ind = 0
    		minDist = float('inf')
    		for k in range(len(centroids)):
    			cur_dist = util.dotProduct(x,x) - 2*util.dotProduct(centroids[k], x) + pre_computed_centroid_dots[k]
    			min_dist = float('inf')
    			if cur_dist < min_dist:
    				assignment = k
    				min_dist = cur_dist
    		return predictor_list[i](x)
        

    	regression_error = boostedtree.evaluatePredictor(predictor, currentTest)
        #regression_error /= len(train_examples)

        # Print the results
    	print ""
    	print "------------------"
    	print "CLUSTERED REGRESSION WITH BOOSTING"
    	print "------------------"
    	print "Leaving out segment: " + str(i)
    	print "Number of centroids: " + str(10)
    	print "Number of examples: " + str(len(train_examples))
    	print "Regression MSE:     " + str(regression_error)
    	print ""

    return predictor_list, centroids, regression_error

trainAndTest()
