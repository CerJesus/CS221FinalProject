"""Beating the Bubble: Cross-Validation

Alexandre Bucquet, Jesus Cervantes, Alex Kim
Python 2.7

This module performs cross-validation on any given predictor.
"""
import math, random
from collections import defaultdict
import numpy  as np
import pandas as pd
from util import dotProduct, increment, lossGradient, featurize, \
        evaluatePredictor, csvAsArray, getCsvHeaders
import regression.py


def crossValidate(predictor, num_folds):
    """Performs k-fold cross validation on a specified predictor function and
    prints the results.

    Args:
        predictor (func): A predictor function.
        num_folds (int): Number of data folds for cross-validation.
    """
    # Import the training data as a numpy array
    train_array = csvAsArray('data/train_updated.csv')

    # Generate a list of (feature vector, value) tuples for the training data
    feature_names = getCsvHeaders('data/train_updated.csv')

    # Convert the training array into ([features], value) example tuples
    train_examples = []
    for i in range(len(train_array)):
        feature_count  = range(len(train_array[i]) - 1)
        feature_values = [train_array[i][j] for j in feature_count]
        feature_vector = featurize(feature_values, feature_names)
        output         = train_array[i][len(train_array[0]) - 1]
        train_examples.append((feature_vector, output))

    # Randomize the order of the example tuples to aid validation
    random.shuffle(train_examples)

    # Validation on each fold
    validation_set_size = len(train_examples) / num_folds
    for fold in range(num_folds):

        # Create training and validation sets
        valdiation_start = fold * validation_set_size
        validation_end   = validation_start + validation_set_size
        validation_set = train_examples[validation_start:validation_end]
        training_set   = train_examples[:validation_start]+train_examples[validation_end:]

        # Train a regression model on the training data and evaluate its mean
        # squared error with the validation set
        tuning_parameter = 1 
        predictor_fn = predictor(train, 1, 0.01, tuning_parameter)
        regression_error = evaluatePredictor(predictor_fn, validation_set)

        # Print the results
        print ""
        print "----------"
        print "REGRESSION"
        print "----------"
        print "Lambda: ", tuning_parameter
        print "Number of examples: ", len(train_examples)
        print "Regression MSE:     ", regression_error
        print ""


if __name__ == "__main__":
    cross_validate(regression.learnRegression, 3)
