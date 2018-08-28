"""Beating the Bubble: Baseline Predictor

Alexandre Bucquet, Jesus Cervantes, Alex Kim
Python 2.7

This module defines and trains a naive baseline predictor. It also evaluates the
predictor's mean squared error after training.
"""
import math
import random
from collections import defaultdict

import numpy  as np
import pandas as pd
from util import evaluatePredictor, csvAsArray


def learnBaseline(np_array):
    """Generates the baseline predictor function.

    Learns the baseline predictor function by finding the median house price in
    the training data.

    Args:
        np_array: A numpy array consisting of training examples.

    Returns:
        A predictor function, which returns the median sale price (int).
    """
    sale_prices = [np_array[i][80] for i in range(len(np_array))]
    np.sort(sale_prices)
    median_sale_price = sale_prices[len(sale_prices) / 2 - 1] / 1000.0
    print median_sale_price
    def baselinePredictor(x):
        return median_sale_price
    return baselinePredictor


def learnOracle(examples):
    """Generates the oracle predictor function.

    Creates a predictor function that simply returns the true house price of
    any given example.

    Args:
        examples: An array consisting of each training examples

    Returns:
        A predictor function, which returns the true sale price (int).        
    """
    examples_map = {}
    for x, y in examples:
        examples_map[x] = y

    def oraclePredictor(x):
        return examples_map[x]

    return oraclePredictor


def trainAndEvaluate():
    """Trains a baseline predictor and prints its mean squared error.
    """
    # Import the training and test data as numpy matrices
    train_array = csvAsArray('data/train.csv')

    # Format the training data as a list of (input, output) tuples
    train_examples = []
    for i in range(len(train_array)):
        input_size = range(len(train_array[i]) - 1)
        input_data = (train_array[i][j] for j in input_size)
        output     = train_array[i][80] / 1000.0
        train_examples.append((input_data, output))

    # Define predictor functions for baseline and oracle
    baseline     = learnBaseline(train_array)
    oracle_train = learnOracle(train_examples)

    # Evaluate mean squared error of predictors
    baseline_error = evaluatePredictor(baseline, train_examples)
    oracle_error   = evaluatePredictor(oracle_train, train_examples)

    # Print the results
    print ""
    print "-------------------"
    print "BASELINE AND ORACLE"
    print "-------------------"
    print "Number of examples:    ", len(train_examples)
    print "Baseline (median) MSE: ", baseline_error
    print "Oracle MSE:            ", oracle_error
    print ""


if __name__ == "__main__":
    trainAndEvaluate()
