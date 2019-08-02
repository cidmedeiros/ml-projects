# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:41:45 2019

@author: cidm
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt

data = pd.read_csv('titanic_data.csv') 

nas = pd.concat([data.isnull().sum()], axis=1, keys=['data']) #pd.concat working as a stack operator
missing_list = list(nas[nas.sum(axis=1) > 0].index)

data = data.drop(columns=['Cabin', 'Name', 'Ticket'])
data.Age = data.Age.fillna(data.Age.mean())
data.Embarked = data.Embarked.fillna(data.Embarked.mode()[0])
data.Pclass = data.Pclass.apply(str)

data = pd.get_dummies(data)

labels = data.pop('Survived')

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)

clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None,
                             min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                             presort=False, random_state=None, splitter='best')

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

###MAX_DEPTH

max_depths = np.linspace(1, 32, 32, endpoint=True) #Return evenly spaced numbers over a specified interval.

train_results = []
test_results = []

for max_depth in max_depths:
   clf = DecisionTreeClassifier(max_depth=max_depth)
   clf.fit(x_train, y_train)
   
   train_pred = clf.predict(x_train)
   
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   # Add auc score to previous train results
   train_results.append(roc_auc)
   
   y_pred = clf.predict(x_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   
   roc_auc = auc(false_positive_rate, true_positive_rate)
   # Add auc score to previous test results
   test_results.append(roc_auc)
   
##PLOT ROC CURVE

line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC Score')
plt.xlabel('Tree Depth')
plt.show()

###MIN_SAMPLES_SPLIT

min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)

train_results = []
test_results = []

for min_samples_split in min_samples_splits:
   clf = DecisionTreeClassifier(min_samples_split=min_samples_split)
   clf.fit(x_train, y_train)
   
   train_pred = clf.predict(x_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   
   y_pred = clf.predict(x_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)

##PLOT ROC CURVE

line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples split')
plt.show()
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
