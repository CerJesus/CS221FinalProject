#!/usr/bin/python

import random
import collections
import math
import sys
import util

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)

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
    assign = [0 for _ in range(len(examples))]

    precomputed_x = [util.dotProduct(examples[i], examples[i]) for i in range(len(examples))]

    for i in range(maxIters):
        print "Progress:", 1.0*i/maxIters * 100, "%"
    	loss = 0
    	means = [{} for _ in range(K)]
    	cluster_count = [0 for _ in range(K)]
    	prev_centroids = centroids
    	prev_assign = assign[:]

    	precomputed_quantities = [util.dotProduct(centroids[i], centroids[i]) for i in range(K)]

    	#loop through the examples to assign
    	for j in range(len(examples)):
    		assign[j] , dist = find_center(j, examples[j], precomputed_x, precomputed_quantities, centroids)
    		loss += dist
    		util.increment(means[assign[j]], 1, examples[j])
    		cluster_count[assign[j]] += 1

    	if assign == prev_assign:
            print loss
            return centroids, assign, loss

    	for index in range(K):
    		divide(means[index], cluster_count[index])
    	centroids = means
        
    	if centroids == prev_centroids:
            print loss
            return centroids, assign, loss

    print "The reconstruction loss is:", loss
    return centroids, assign, loss
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
    train_array = util.csvAsArray('train_updated.csv')
    test_array  = util.csvAsArray('test.csv')

    # Generate a list of (feature vector, value) tuples for the training data
    feature_names = util.getCsvHeaders('train_updated.csv')
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

    (centroids, assign, loss) = kmeans(k_examples, 15, 500)
#print centroids, assign, loss

    # Print the results
    print ""
    print "------------------"
    print "K-MEANS WITH " + str(15) + " CENTROIDS"
    print "------------------"
    print "Number of examples: " + str(len(k_examples))
    print "Reconstruction loss: " + str(loss)
    print "Average reconstruction loss: " + str(loss/len(k_examples))
    print ""

trainAndTest()
