#!/usr/bin/env python

## import libraries to be used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d # for 3D plots
import scikit-learn
from sklearn.datasets import load_diabetes # repurposing plotting code week 8 homework

diabetes = load_diabetes(return_X_y = False)

data = diabetes.data
headers = diabetes.feature_names + 'Y'
target = diabetes.target

## plot and save individual histograms for each feature/variable
for i in range(0, len(headers)):
    plt.hist(data[i])

    plt.xlabel(headers[i])
    plt.ylabel('Count')

    hname = 'hist_' + i + '.png'
    plt.savefig(hname)
    plt.clf()

## correlation martrix (seaborn heatmap)
f, ax = plt.subplots(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(100, 220, as_cmap=True),
            square=True, ax=ax)

plt.savefig('diabetes_corr_matrix.png')
plt.clf()

## 2 feature/variable scatter plots (Y vs. X)
for i in range(0, len(headers)):
    xvar = headers[i]

    for j in range(0, len(headers)-1):
        if i != j:
            yvar = headers[j]
            plt.scatter(data[i], data[j])

            plt.xlabel(xvar)
            plt.ylabel(yvar)

            sname = 'scatter_' + yvar + '_vs_' + xvar + '.png'
            plt.savefig(sname)
            plt.clf()

## 4 feature scatter plot
# best Y predicting features based on correlation matrix (ie. highest absolute r coefficient): BMI, BP, S3 (-ve), S5
ydata = target

for feature in [2, 3, 6, 8]:
    plt.scatter(data[feature], ydata, label=feature)

plt.xlabel('Input Feature Scale')
plt.ylabel('Y')

plt.legend(loc='right', bbox_to_anchor=(1.2, 0.5))
plt.grid(True)

plt.savefig('FourFeatureScatter_best4predictors.png', bbox_inches='tight')
plt.clf()

# best Y predicting features based on correlation matrix (ie. greatest +ve r coefficient): BMI, BP, S5
for feature in [2, 3, 8]:
    plt.scatter(data[feature], ydata, label=feature)

plt.xlabel('Input Feature Scale')
plt.ylabel('Y')

plt.legend()
plt.grid(True)

plt.savefig('FourFeatureScatter_best+vePredictors.png')
plt.clf()

## 3D line plots involving SEX variable

xdata = data[1]
zdata = target # plot the outcome variable of interest on vertical axis

for i in range(0, len(headers)-3):
    if i != 1:  # SEX column
        # set up 3D plot figure and axes
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ydata = data[i]

        ax.scatter3D(xdata,ydata,zdata,c=ydata, marker='x')

        if i == 0:
            x2header = headers[i+2]
            y2header = headers[i+3]
            ax.plot3D(data[i+2],data[i+3], zdata, c='gray')
        else:
            x2header = headers[i+1]
            y2header = headers[i+2]
            ax.plot3D(data[i+1],data[i+2], zdata, c='gray')

        ax.set_xlabel('SEX and '+ x2header)
        ax.set_ylabel(headers[i] + ' and ' + y2header)
        ax.set_zlabel('Y')

        plt.savefig('SEX, '+ headers[i] + ', ' + x2header + ', ' + y2header + ' for Y.png')
        plt.clf()

## 3D plots without SEX variable
headers.remove('sex')

for i in range(0, len(headers)-2):
    x1data = data[i]

    for j in range(1, len(headers)-1):
        if i < j:
            y1data = data[j]

            fig = plt.figure()
            ax = plt.axes(projection='3d')

            ax.scatter3D(x1data, y1data, zdata)

            k = j+1

            if k == 9:
                k=0

            x2data = data[k]

            ax.scatter3D(x2data, 0, zdata)

            ax.set_xlabel(headers[i])
            ax.set_ylabel(headers[j])
            ax.set_zlabel('Y')

            plt.savefig(headers[i] + ', ' + headers[j] + ', ' + headers[k] + ', Y.png')
            plt.clf()

for i in range(0, len(headers)-2):
    x1data = data[i]

    for j in range(1, len(headers)-1):
        if i < j:
            y1data = data[j]

            fig = plt.figure()
            ax = plt.axes(projection='3d')

            ax.scatter3D(x1data, y1data, zdata)

            k = j+1

            if k == 9:
                k=1

            x2data = data[k]

            ax.scatter3D(x2data, 0, zdata)

            ax.set_xlabel(headers[i])
            ax.set_ylabel(headers[j])
            ax.set_zlabel('Y')

            plt.savefig(headers[i] + ', ' + headers[j] + ', ' + headers[k] + ', Y.png')
            plt.clf()
