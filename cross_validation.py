"""
Beating the Bubble: Housing Prices in Ames, Iowa

Filename: cross_validation.py
Authors:  Alexandre Bucquet, Jesus Cervantes, Alex Kim
Python 2.7

DESCRIPTION
This script performs cross-validation on any given predictor.
"""

import math, random
from collections import defaultdict
import numpy  as np
import pandas as pd
from util import dotProduct, increment, lossGradient, featurize, \
        evaluatePredictor, csvAsArray, getCsvHeaders

# CROSS-VALIDATION FUNCTIONS ---------------------------------------------------


