"""Beating the Bubble: Utilities

Alexandre Bucquet, Jesus Cervantes, Alex Kim
Python 2.7

This module defines common utility functions.
"""

import math, random
from collections import defaultdict
import numpy  as np
import pandas as pd


# VECTOR FUNCTIONS -------------------------------------------------------------

def dotProduct(vec1, vec2):
    """Compute the dot product of two vectors.

    Args:
        vec1 (dict): A sparse vector.
        vec2 (dict): A sparse vector.

    Returns:
        The dot product of two vectors (float).
    """
    dot_product = 0
    for key, value in vec1.iteritems():
        if key in vec2.keys():
            dot_product += vec1[key] * vec2[key]
    return dot_product


def increment(vec1, scale, vec2):
    """Increment one vector by the quantity of a scaled vector.

    This function essentially performs: vec1 += scale * vec2

    Args:
        vec1 (dict): A sparse vector, which gets incremented.
        scale (float): A scalar for vec2.
        vec2 (dict): A sparse vector, which is used to increment vec1.
    """
    for key, value in vec2.items():
        if key in vec1.keys():
            vec1[key] += scale * vec2[key]
        else:
            vec1[key] = scale * vec2[key]


# MODELING FUNCTIONS -----------------------------------------------------------

def lossGradient(features, weights, true_value):
    """Compute the value of the training loss gradient (with respect to the
    weight vector) at a specific example.

    Args:
        features (dict): A sparse vector of feature values.
        weights (dict): A sparse vector of feature weights.
        true_value (int): The true value of an example.

    Returns:
        A sparse vector (dict) representing the gradient value.
    """
    gradient = {}
    scale = 2 * (dotProduct(features, weights) - true_value)
    increment(gradient, scale, features)
    return gradient


def featurize(feature_values, feature_names):
    """For a feature vector for one example, convert enumerated features into
    corresponding indicator features.

    Args:
        feature_values (list): A vector of the feature values for one example.
        feature_names (list): A vector of the names of each feature.

    Returns:
        A feature vector (dict) whose enumerated feautres are now multiple
        corresponding indicator features.
    """
    features = defaultdict(int)
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


def evaluatePredictor(predictor, examples):
    """Compute the mean squared error of a predictor function.

    Args:
        predictor (func): A predictor function.
        examples (array): An array where each row is an example.

    Returns:
        The mean squared error (float) of the predictor.
    """
    error = 0.0
    for input, output in examples:
        error += (predictor(input) - output) ** 2
    error /= len(examples)
    return error


# DATA IMPORT ------------------------------------------------------------------

def csvAsArray(data_file):
    """Create a numpy array of data from a CSV file.

    Args:
        data_file (string): The filename of a CSV file.

    Returns:
        A numpy array of the CSV data.
    """
    data = pd.read_csv(data_file)
    data_array = data.values
    return data_array


def getCsvHeaders(data_file):
    """Extract the headers of a CSV file.

    Args:
        data_file (string): The filename of a CSV file.

    Returns:
        A list of the headers of the CSV file.
    """
    data = pd.read_csv(data_file)
    headers = data.columns.values[1:]
    return headers
