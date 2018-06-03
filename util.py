"""
Beating the Bubble: Housing Prices in Ames, Iowa

Filename: util.py
Authors:  Alexandre Bucquet, Jesus Cervantes, Alex Kim
Python 2.7

DESCRIPTION
Common functions
"""

import math, random
from collections import defaultdict
import numpy  as np
import pandas as pd

# VECTOR FUNCTIONS -------------------------------------------------------------

# DOT PRODUCT: Return the dot product of two numpy vectors
def dotProduct(vec1, vec2):
    dot_product = 0
    for key, value in vec1.iteritems():
        if key in vec2.keys():
            dot_product += vec1[key] * vec2[key]
    return dot_product

# INCREMENT: Increment a vector (vec1 += scale * vec2)
def increment(vec1, scale, vec2):
    for key, value in vec2.items():
        if key in vec1.keys():
            vec1[key] += scale * vec2[key]
        else:
            vec1[key] = scale * vec2[key]

# MODELING FUNCTIONS -----------------------------------------------------------

# LOSS GRADIENT: Return the gradient of the training loss with respect to the
# weight vector for a given example (features, true_value)
def lossGradient(features, weights, true_value):
    gradient = {}
    scale = 2 * (dotProduct(features, weights) - true_value)
    increment(gradient, scale, features)
    return gradient

# FEATURIZE: Given a feature vector, return an updated feature vector (in the
# form of a dict). Turns enumerated string features into unique indicator
# features.
def featurize(feature_values, feature_names):
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

# PREDICTOR EVALUATION: Compute the mean squared error of a predictor function
# given a matrix of examples
def evaluatePredictor(predictor, examples):
    error = 0.0
    for input, output in examples:
        error += (predictor(input) - output) ** 2
    error /= len(examples)
    return error

# DATA IMPORT ------------------------------------------------------------------

# IMPORT DATA: Takes the filename of a csv and returns the data as a numpy array
def csvAsArray(data_file):
    data = pd.read_csv(data_file)
    data_array = data.values
    return data_array

# HEADERS: Takes the filename of a csv and returns its headers
def getCsvHeaders(data_file):
    data = pd.read_csv(data_file)
    headers = data.columns.values[1:]
    return headers
