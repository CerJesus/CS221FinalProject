#!/usr/bin/python

import random
import collections
import math
import sys
import util
import numpy as np
from boostedtree import lossGradient, learnBoostedRegression
import cPickle as pickle

NUM_B_TREES = 5

NUM_CLUSTERS = 15

NUM_SPLITS = 10

def kmeans(full_examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    examples = [full_examples[j][0] for j in range(len(full_examples))]
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
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE


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
    train_array = util.csvAsArray('data/train_updated.csv')
    test_array  = util.csvAsArray('data/test.csv')
    # Generate a list of (feature vector, value) tuples for the training data
    feature_names = util.getCsvHeaders('data/train_updated.csv')
    train_examples = []
    k_examples = []
    for i in range(len(train_array)):
        feature_count  = range(len(train_array[i]) - 1)
        feature_values = [train_array[i][j] for j in feature_count]
        feature_vector = featurize(feature_values, feature_names)
        output         = train_array[i][len(train_array[0]) - 1]
        train_examples.append((feature_vector, output))
        k_examples.append(feature_vector)
    # Train a k-means model on the training data and evaluate its mean
    # squared error with the test data

    random.shuffle(train_examples)
    for i in range(0, NUM_SPLITS, 2):
    	startTest = i*len(train_examples)/NUM_SPLITS
    	endTest = (i+1)*len(train_examples)/NUM_SPLITS
    	currentTrainExamples = train_examples[0:startTest] + train_examples[endTest:len(train_examples)]
        (centroids, assign, loss, loss_list, centroid_vals) = kmeans(currentTrainExamples, NUM_CLUSTERS, 500)

        currentBoostedExamples = [(currentTrainExamples[ind][0],loss_list[ind]) for ind in range(len(currentTrainExamples))]

        boostedRegPredictor = learnBoostedRegression(currentBoostedExamples, 500, \
                0.00000000001, num_trees=NUM_B_TREES)

        pre_computed_centroid_dots = [util.dotProduct(centroids[ind],centroids[ind]) for ind in range(NUM_CLUSTERS)]
        def kmeanspredictor(x):
            assignment = 0
            min_dist = 1000000
            for j in range(NUM_CLUSTERS):
            	cur_dist = util.dotProduct(x,x) - 2*util.dotProduct(centroids[j], x) + pre_computed_centroid_dots[j]
            	if cur_dist < min_dist:
            		assignment = j
            		min_dist = cur_dist
            return centroid_vals[assignment]

        def boostedKPredictor(x):
            return kmeanspredictor(x) + boostedRegPredictor(x)

        print "leaving out the", (i+1), "th segment of the data, the validation error for the regression is:", util.evaluatePredictor(boostedKPredictor, train_examples[startTest:endTest])

    # (centroids, assign, loss, loss_list, centroid_vals) = kmeans(train_examples, NUM_CLUSTERS, 500)
    # pickle.dump((centroids, assign, loss, loss_list, centroid_vals), open("centroids.p","wb"))
    # print centroid_vals
    #(centroids, assign, loss, loss_list, centroid_vals) = pickle.load(open("centroids.p","rb"))

    #boostedExamples = [(train_examples[j][0],loss_list[j]) for j in range(len(k_examples))]

    # list_weights, num_trees, num_iters = pickle.load(open("kmeansboosted.p","rb"))
    # print num_trees
    # print num_iters
    # def boostedRegPredictor(x):
    #     return sum(util.dotProduct(x, curWeight) for curWeight in list_weights)

    # boostedRegPredictor = learnBoostedRegression(boostedExamples, 500, \
    #         0.00000000001, num_trees=NUM_B_TREES)

    # pre_computed_centroid_dots = [util.dotProduct(centroids[i],centroids[i]) for i in range(NUM_CLUSTERS)]
    # def kmeanspredictor(x):
    #     assignment = 0
    # 	min_dist = 1,000
    # 	for i in range(NUM_CLUSTERS):
    # 		cur_dist = util.dotProduct(x,x) - 2*util.dotProduct(centroids[i], x) + pre_computed_centroid_dots[i]
    # 		if cur_dist < min_dist:
    # 			assignment = i
    # 			min_dist = cur_dist
    #     return centroid_vals[assignment]
    #
    # def boostedKPredictor(x):
    #     return kmeanspredictor(x) + boostedRegPredictor(x)
    #
    # avgError = util.evaluatePredictor(boostedKPredictor,train_examples)
    #print centroids, assign, loss

    # Print the results
    # print ""
    # print "------------------"
    # print "K-MEANS WITH " + str(NUM_CLUSTERS) + " CENTROIDS"
    # print "------------------"
    # print "Number of examples: " + str(len(k_examples))
    # print "Reconstruction loss: " + str(loss)
    # print "Average reconstruction loss: " + str(loss/len(k_examples))
    # print "Boosted average: " + str(avgError)
    # print ""

#trainAndTest()
