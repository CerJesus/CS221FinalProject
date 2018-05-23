#This is our logistic regression file
import math, random
from collections import defaultdict
import numpy
import pandas as pd
import boostedtree
import regression

## Processing training data
file_train = 'train_updated.csv'
data_train = pd.read_csv(file_train)
col_names = data_train.columns.tolist()[1:]

#print(data_train.head())
train_array = data_train.as_matrix(columns=None)

train_examples = [ ( [train_array[i][j] for j in range(len(train_array[i]) - 1) ], train_array[i][len(train_array[0]) - 1]) for i in range(len(train_array))]

# split the data, train and test
numSplits = 10
for i in range(numSplits):
	startTest = i*len(train_examples)/10
	endTest = (i+1)*len(train_examples)/10
	currentTrainExamples = train_examples[0:startTest] + train_examples[endTest:len(train_examples)]
	logisticPredictor = boostedtree.learnRegression(currentTrainExamples, 10, 0.00000000001)
	print "leaving out the", (i+1), "th segment of the data, the testing error for the regression is:", boostedtree.evaluatePredictor(logisticPredictor, train_examples[startTest:endTest])

#test_examples = ( ( (test_array[i][j] for j in range(len(test_array[i]) - 1) ), test_array[i][79]) for i in range(len(test_array)))

#logisticPredictor = learnRegression(train_examples, 10, 0.00000000001)
#print "the training error for the regression is:", evaluatePredictor(logisticPredictor, train_examples)
