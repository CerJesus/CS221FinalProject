"""Beating the Bubble: Standard Regression

Alexandre Bucquet, Jesus Cervantes, Alex Kim
Python 2.7

This module defines a standard linear regression predictor and evaluates its
mean squared error.
"""
import math
import random
from collections import defaultdict
import numpy  as np
import pandas as pd
from util import dotProduct, increment, lossGradient, featurize, \
        evaluatePredictor, csvAsArray, getCsvHeaders


def lassoLossGradient(features, weights, true_value, tuning_parameter):
    """Computes the value of the training loss gradient (with respect to the
    weight vector) at a specific example.

    Training loss includes a lasso (L1) regularization term.

    Args:
        features (dict): A sparse vector of feature values.
        weights (dict): A sparse vector of feature weights.
        true_value (int): The true value of an example.
        tuning_parameter (double): Coefficient of the lasso regularization term.

    Returns:
        A sparse vector (dict) representing the gradient value.
    """
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


def regularizationLossGradient(features, weights, true_value, tuning_parameter):
    """Computes the value of the training loss gradient (with respect to the
    weight vector) at a specific example.

    Training loss includes a ridge (L2) regularization term.

    Args:
        features (dict): A sparse vector of feature values.
        weights (dict): A sparse vector of feature weights.
        true_value (int): The true value of an example.
        tuning_parameter (double): Coefficient of the ridge regularization term.

    Returns:
        A sparse vector (dict) representing the gradient value.
    """
    # Standard squared loss
    gradient = {}
    scale =  2 * (dotProduct(features, weights) - true_value)

    # Regularization term: add gradient of the regularization term to the
    # scaling factor (i.e. add gradient of |tuning_parameter| *
    # (2-norm of weights)^2 
    increment(gradient, tuning_parameter, weights)
    increment(gradient, scale, features)
    return gradient


def learnRegression(examples, numIters, stepSize, tuning_parameter):
    """Learns linear regression weights and generates a predictor function.

    Args:
        examples: An array of training examples.
        numIters (int): Number of training iterations.
        stepSize(double): Stochastic gradient descent step size.
        tuning_parameter (double): Tuning parameter for the loss function.
        
    Returns:
        A predictor function that outputs a price (int) given a single input
        tuple.
    """
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


def trainAndEvaluate():
    """Trains a linear regression predictor and prints its mean squared error.
    """
    # Import the training data as a numpy array
    train_array = csvAsArray('data/train_updated.csv')

    # Generate a list of (feature vector, value) tuples for the training data
    feature_names = getCsvHeaders('data/train_updated.csv')
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
        regressionPredictor = learnRegression(train, 500, 0.00000000001,
                tuning_parameter)
        regression_error = evaluatePredictor(regressionPredictor, test)

        # Print the results
        print ""
        print "----------"
        print "REGRESSION"
        print "----------"
        print "Lambda (lasso): ", tuning_parameter
        print "Number of examples: ", len(train_examples)
        print "Regression MSE:     ", regression_error
        print ""


if __name__ == "__main__":
        trainAndEvaluate()
