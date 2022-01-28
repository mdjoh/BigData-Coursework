#!/usr/bin/env python

## import libraries to be used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes # repurposing plotting code week 8 homework
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

## load diabetes dataset and assign data to variables
diabetes = load_diabetes(return_X_y = False)

data = diabetes.data
headers = diabetes.feature_names
target = diabetes.target
target_header = 'Y'

### Decision Tree ###
clf = DecisionTreeRegressor().fit(data, target)
predicted = clf.predict(data)
expected = target

# plot predicted vs expected
plt.scatter(expected, predicted)
plt.xlabel('Features')
plt.ylabel('Predicted ' + target_header)
plt.savefig('DecisionTreeOutput.png')
plt.clf()

## Regression steps:
# 1. instantiate the model
# 2. fit the model with x and y training data
# 3. evaluate the results of the model
# 4. plot the scatter plot of predicted vs. expected

# Split the data and target into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, target)

### Linear Regression ###
clf = LinearRegression()
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
expected = y_test
print("RMS: %s" % np.sqrt(np.mean((predicted - expected)**2)))

# print the model score
clf.score(X_test, y_test)

# plot predicted vs expected
plt.scatter(expected, predicted)
plt.xlabel('Features')
plt.ylabel('Predicted ' + target_header)
plt.savefig('LinearRegOutput.png')
plt.clf()

### Gradient Boosting Tree Regression ###
clf = GradientBoostingRegressor()
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
expected = y_test
print("RMS: %s" % np.sqrt(np.mean((predicted - expected)**2)))

# print the model score
clf.score(X_test, y_test)

# plot predicted vs expected
plt.scatter(expected, predicted)
plt.xlabel('Features')
plt.ylabel('Predicted ' + target_header)
plt.savefig('GradientRegOutput.png')
plt.clf()

######################## PREDICTIONS WITHOUT CATEGORICAL FEATURE 'SEX' #################################

# dataset without sex (ie. a categorical feature) column which is the 2nd column in original dataset
nsdata = np.delete(data, 1, 1)

### Decision Tree ###
clf = DecisionTreeRegressor().fit(nsdata, target)
predicted = clf.predict(nsdata)
expected = target

# plot predicted vs expected
plt.scatter(expected, predicted)
plt.xlabel('Features')
plt.ylabel('Predicted ' + target_header)
plt.savefig('DecisionTreeOutput_noSex.png')
plt.clf()

# Split the data and target into train and test sets
X_train, X_test, y_train, y_test = train_test_split(nsdata, target)

### Linear Regression ###
clf = LinearRegression()
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
expected = y_test
print("RMS: %s" % np.sqrt(np.mean((predicted - expected)**2)))

# print the model score
clf.score(X_test, y_test)

# plot predicted vs expected
plt.scatter(expected, predicted)
plt.xlabel('Features')
plt.ylabel('Predicted ' + target_header)
plt.savefig('LinearRegOutput_noSex.png')
plt.clf()

### Gradient Boosting Tree Regression ###
clf = GradientBoostingRegressor()
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
expected = y_test
print("RMS: %s" % np.sqrt(np.mean((predicted - expected)**2)))

# print the model score
clf.score(X_test, y_test)

# plot predicted vs expected
plt.scatter(expected, predicted)
plt.xlabel('Features')
plt.ylabel('Predicted ' + target_header)
plt.savefig('GradientRegOutput_noSex.png')
plt.clf()
