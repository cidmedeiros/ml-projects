# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:21:36 2019

@author: cidm
"""
import pickle
import mnist_functions as f
import numpy as np
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

with open('mnist.pkl', 'rb') as handler:
    mnist = pickle.load(handler)

x,y = mnist['data'], mnist['target']

some_digit = x[36000]
f.image_show(some_digit) #example of displaying an image from the dataset

"""
This dataset is actually already split into training set and test set 6-10.
Let's also shuffle the training set. This will guarantee that all cross-validation
folds will be similar.
"""
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

"""
Let's first try to id only the handwritten 5s
"""   
y_train_5, y_test_5 = (y_train == 5), (y_test == 5)#replace the original target array with a new one stating wether the algarism is a 5 or not

sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
sgd_clf.fit(x_train, y_train_5)
y_train_pred = sgd_clf.predict(x_train)
"""
Let's test our classifier
"""
sgd_clf.predict([some_digit])


"""
Implementing a cross-validation model.
The following code does roughly the same thing cross_val_score function.
"""
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(x_train, y_train_5):
    clone_clf = clone(sgd_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train_5[train_index]
    x_test_fold = x_train[test_index]
    y_test_fold = y_train_5[test_index]
    
    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_fold)
    n_correct = sum(y_pred==y_test_fold)
    
    print(n_correct/len(y_pred))

from sklearn.model_selection import cross_val_score, cross_val_predict

cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring='accuracy')

"""
Off-course accuracy is a bad metric for skewed bynary classification tasks.
Let's take a look at the confusion matrix for our predictor so far.
"""
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)

"""
F1-Score
"""
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
f1_score(y_train_5, y_train_pred)

"""
Accessing and playing with thresholds for the precision recall tradeoff
"""
y_scores = sgd_clf.decision_function([some_digit])
threshold = 0
y_some_digit_pred = (y_scores > threshold)
threshold = 200000
y_some_digit_pred = (y_scores > threshold)

"""
The problem with the method above is that we get only one score at a time.
cross_val_predict() and precision_recall_curve()
"""
y_scores = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3, method='decision_function')

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
f.plot_precision_recall_threshold(precisions, recalls, thresholds)
f.plot_precision_recall(precisions, recalls)

"""
Now let's say you need a 90% precision classifier. So, after analizying the plots,
you realize your model's threshold has to be set at 190950.
"""

y_train_pred_90 = (y_scores > 190950)

precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
f.plot_roc_curve(fpr, tpr)#plots roc curve

roc_auc_score(y_train_5, y_scores)

"""
Let's compare ROC curves for different classifiers: SGD, RandomForest
In some cases due to the nature of the algorithm, the classifier doesn't
use scores but probability instead. That's the case for Random Forests.
In those cases, we'll use the positive class probabilities as the scores.
"""
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state = 42)
y_probas_forest = cross_val_predict(forest_clf, x_train, y_train_5, cv=3, method='predict_proba')
y_scores_forest = y_probas_forest[:, 1] #proba of positive class

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, 'b:', label='SGD')
f.plot_roc_curve(fpr_forest, tpr_forest, label='Random Forest')
plt.legend(loc='lower right')
plt.show()

forest_clf.fit(x_train, y_train_5)
y_pred_forest = forest_clf.predict(x_train)

y_pred_forest = (y_scores_forest > 0.8)

precision_score(y_train_5, y_pred_forest)
recall_score(y_train_5, y_pred_forest)








































