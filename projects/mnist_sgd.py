# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:21:36 2019

@author: cidm
"""
import pickle
import mnist_functions as f
import numpy as np
from sklearn.linear_model import SGDClassifier

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
y_train, y_test = (y_train == 5), (y_test == 5)#replace the original target array with a new one stating wether the algarism is a 5 or not

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train)

"""
Let's test our classifier
"""
sgd_clf.predict([some_digit])
