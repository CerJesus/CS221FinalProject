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

# MODEL TESTING FUNCTIONS ------------------------------------------------------

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
