"""Beating the Bubble: Clustering on Neighborhoods

Alexandre Bucquet, Jesus Cervantes, Alex Kim
Python 2.7

This module runs the k-means algorithm on neighborhoods in order to determine
which ones are most similar.
"""
import random
import collections
import math
import sys
import util
import numpy as np
from boostedtree import lossGradient, learnBoostedRegression
import cPickle as pickle


MAX_NUM_CLUSTERS = 16
STEP_SIZE = 2


def kmeans(full_examples, K, maxIters):
    """Performs k-means clustering on neighborhoods.

    Args:
        full_examples (list): List of examples, each example is a string-to-
            double dict representing a sparse vector.
        K (int): The number of desired clusters. Assume that 0 < K <= |examples|.
        maxIters (int): The maximum number of iterations to run. 

    Returns:
        A tuple of the form (length K list of cluster centroids, list of
        assignments (i.e. if examples[i] belongs to centers[j], then
        assignments[i] = j), final reconstruction loss)
    """
    examples = full_examples
    def find_center(ex_index, example, precomputed_x, precomputed_quantities, centroids):
    	assign = 0
    	min_dist = 1,000
    	for i in range(K):
    		cur_dist = precomputed_x[ex_index] - 2*util.dotProduct(centroids[i], example) + precomputed_quantities[i]
    		if cur_dist < min_dist:
    			assign = i
    			min_dist = cur_dist
    	return assign, min_dist

    def divide(vec, scale):
    	for k, v in vec.items():
    		vec[k] = 1.0 * v/scale

    rand_list = random.sample(xrange(len(examples)), K)
    centroids = [examples[i] for i in rand_list]
    centroid_vals = [full_examples[i][1] for i in rand_list]
    assign = [0 for _ in range(len(examples))]
    loss_list = [0 for _ in range(len(examples))]
    precomputed_x = [util.dotProduct(examples[i], examples[i]) for i in range(len(examples))]

    for i in range(maxIters):
        print "Progress:", 1.0*i/maxIters * 100, "%"
    	loss = 0
    	means = [{} for _ in range(K)]
        val_means = [0 for _ in range(K)]
    	cluster_count = [0 for _ in range(K)]
    	prev_centroids = centroids
    	prev_assign = assign[:]

    	precomputed_quantities = [util.dotProduct(centroids[i], centroids[i]) for i in range(K)]

    	#loop through the examples to assign
    	for j in range(len(examples)):
            assign[j] , dist = find_center(j, examples[j], precomputed_x, precomputed_quantities, centroids)
            util.increment(means[assign[j]], 1, examples[j])
            val_means[assign[j]] += full_examples[j][1]
            cluster_count[assign[j]] += 1

            loss_list[j] = (full_examples[j][1] - centroid_vals[assign[j]])
            loss += dist
        print "LOSS: " + str(loss)
    	if assign == prev_assign:
            print loss
            return centroids, assign, loss, loss_list, centroid_vals

    	for index in range(K):
            divide(means[index], cluster_count[index])
            val_means[index] = val_means[index]/cluster_count[index]

        centroids = means
        centroid_vals = val_means

    	if centroids == prev_centroids:
            print loss
            return centroids, assign, loss, loss_list, centroid_vals

    print "The reconstruction loss is:", loss
    return centroids, assign, loss, loss_list, centroid_vals

# FEATURIZE: Given a feature vector, return an updated feature vector (in the
# form of a dict). Turns enumerated string features into unique indicator
# features.
def featurize(feature_values, feature_names):
    features = collections.defaultdict(int)
    print ""
    for i in range(len(feature_values)):

        # Case 1: string -> indicator
        if type(feature_values[i]) == str:
            features[feature_names[i] + feature_values[i]] = 1

        # Case 2: no value -> NA indicator
        elif math.isnan(feature_values[i]):
            features[feature_names[i] + 'NA'] = 1

        # Case 3: already an int -> no change
        else:
            features[feature_names[i]] = feature_values[i]

    return features

def trainAndTest():

    # Import the training and test data as numpy arrays
    train_array = util.csvAsArray('data/neighborhood_data_final_w_loc.csv')

    # Generate a list of (feature vector, value) tuples for the training data
    feature_names = util.getCsvHeaders('data/neighborhood_data_final_w_loc.csv')

    train_examples = []
    names = []
    for i in range(len(train_array)):
        feature_count  = range(2, len(train_array[i]))
        feature_values = [train_array[i][j] for j in feature_count]
        feature_vector = featurize(feature_values, feature_names[1:])
        train_examples.append(feature_vector)
        names.append(train_array[i][1])
       

    # Train a k-means model on the training data and evaluate its mean
    # squared error with the test data

    for i in range(STEP_SIZE, MAX_NUM_CLUSTERS + 1, STEP_SIZE):
        (centroids, assign, loss, loss_list, centroid_vals) = kmeans(full_examples=train_examples, K=i, maxIters=500)
        filename = "neighborhood_centroids" + str(i) + ".p"
        pickle.dump((centroids, assign, loss, loss_list, centroid_vals, names), open(os.path.join("neighborhood_centroids", filename), "wb"))
        print names, assign

trainAndTest()
