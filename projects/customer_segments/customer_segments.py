# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:00:45 2019

@author: cidm
"""

import numpy as np
import pandas as pd
import visuals as vs
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('customers.csv')
data.drop(['Region', 'Channel'], axis=1, inplace=True)

data_stats = data.describe().T

indices = [20,128,250,380]
samples = pd.DataFrame(data.loc[indices], columns=data.keys()).reset_index(drop=True)

"""
I would expect markets to be establishments with spending on Fresh products above the mean as well as higher
than any other product category. The first and the fourth sample fill in that expectation, so I would argue
these two samples as markets establishments.

I think cafes are the kind of establishments where milk would play a major role in cooking baked goods. With
that in mind, the milk's spending for the second sample really stands out, so I would label it as a cafe.

The third sample doesn't hold any particular pattern on its spendings, they were all similar and below average.
Therefore, I think this might be a general-goods oriented small around the corner retailer.

Comparing specific observations values against measures of central tendency helps to get a more informed sense
about where these observations might be located in the overall classes of the dataset.
"""

# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data.drop(['Delicatessen'], axis=1)
y = data.Frozen
# TODO: Split the data into training and testing sets(0.25) using the given feature as the target
# Set a random state.
X_train, X_test, y_train, y_test = train_test_split(new_data, y, test_size = 0.25, random_state=42)

# TODO: Create a decision tree regressor and fit it to the training set
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

# TODO: Report the score of the prediction using the testing set
score = regressor.score(X_test, y_test)