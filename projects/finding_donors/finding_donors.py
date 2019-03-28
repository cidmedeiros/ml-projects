# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:22:41 2019

@author: cidm
"""

import pandas as pd
import visuals as vs
import numpy as np

data = pd.read_csv('census.csv')
vs.distribution(data).savefig('skewed_graph.jpg')

income_raw = data['income']
features_raw = data.drop('income', axis = 1)

skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed = True).savefig('log_transformed_graph.png')

#Scaling the features
from sklearn.preprocessing import MinMaxScaler
# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() #default=(0, 1)

numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

#An example of a record with scaling applied
scaling_example = features_log_minmax_transform.head(5)

#One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
features_final = pd.get_dummies(features_log_minmax_transform)

#Encode the 'income_raw' data to numerical values
income = np.where(income_raw == '<=50K', 0, 1)

#Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

#Import train_test_split
from sklearn.model_selection import train_test_split

#Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, income, test_size = 0.2, random_state = 0)

#Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

#Compile all the preprocessed data
df = pd.DataFrame(features_final)
df['income'] = income

#TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
#encoded to numerical values done in the data preprocessing step.
#FP = income.count() - TP-Specific to the naive case

#TN = 0 # No predicted negatives in the naive case
#FN = 0 # No predicted negatives in the naive case

#Calculate accuracy, precision and recall
naive_pred = np.ones((len(income),), dtype=int)
tp = sum([1 if p == c else 0 for p, c in zip(naive_pred,income)])
fn = 0
accuracy = tp/len(income)
recall = tp/(tp+fn)
precision = tp/(tp+(len(income)-tp))

#Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
beta = 0.5
fscore = (1+beta**2)*((precision*recall)/((beta**2*precision)+recall))
 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))

#Import the three supervised learning models from sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#Initialize the three models
clf_A = DecisionTreeClassifier(random_state=42)
clf_B = SVC(random_state=42)
clf_C = KNeighborsClassifier()

#Calculate the number of samples for 1%, 10%, and 100% of the training data
#samples_100 is the entire training set i.e. len(y_train)
#samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
#samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
samples_100 = len(y_train)
samples_10 = int(0.1*samples_100)
samples_1 = int(0.01*samples_100)

#Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = vs.train_predict(clf, samples, X_train, y_train, X_test, y_test)

#Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore).savefig('performance.jpg')

###GridSearchCV
#Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#Initialize the classifier
clf = DecisionTreeClassifier()

#Create the parameters list you wish to tune, using a dictionary if needed.
#parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = {}

#Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta=0.5)

#Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scoring = scorer)

#Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

#Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

#Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print(best_clf.get_params)
