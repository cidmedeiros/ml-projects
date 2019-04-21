# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:58:24 2019

@author: cidmedeiros
"""
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris()

sample = iris.data[:10]
targets = iris.target

# Hierarchical clustering
# Ward is the default linkage algorithm
ward = AgglomerativeClustering(n_clusters = 3)
ward_pred = ward.fit_predict(iris.data)

ward_ar_score = adjusted_rand_score(iris.target, ward_pred)

# Hierarchical clustering using complete linkage
complete = AgglomerativeClustering(n_clusters=3, linkage = 'complete')
complete_pred = complete.fit_predict(iris.data)

complete_ar_score = adjusted_rand_score(iris.target, complete_pred)

# Hierarchical clustering using average linkage
avg = AgglomerativeClustering(n_clusters=3, linkage = 'average')
avg_pred = avg.fit_predict(iris.data)

avg_ar_score = adjusted_rand_score(iris.target, avg_pred)

print( "Scores: \nWard:", ward_ar_score,"\nComplete: ", complete_ar_score, "\nAverage: ", avg_ar_score)

"""
The forth column has smaller values than the rest of the columns, and so its variance counts for
less in the clustering process (since clustering is based on distance). Let us normalize the
dataset so that each dimension lies between 0 and 1, so they have equal weight in the clustering
process.
This is done by subtracting the minimum from each column then dividing the difference by the range.
Sklearn provides us with a useful utility called preprocessing.normalize() that can do that for us
"""
normalized_X = preprocessing.normalize(iris.data)
normalized_X[:10]

ward = AgglomerativeClustering(n_clusters=3)
ward_pred = ward.fit_predict(normalized_X)

complete = AgglomerativeClustering(n_clusters=3, linkage="complete")
complete_pred = complete.fit_predict(normalized_X)

avg = AgglomerativeClustering(n_clusters=3, linkage="average")
avg_pred = avg.fit_predict(normalized_X)


ward_ar_score = adjusted_rand_score(iris.target, ward_pred)
complete_ar_score = adjusted_rand_score(iris.target, complete_pred)
avg_ar_score = adjusted_rand_score(iris.target, avg_pred)

print( "Scores: \nWard:", ward_ar_score,"\nComplete: ", complete_ar_score, "\nAverage: ", avg_ar_score)

#Dendrograms Plotting
linkage_type = 'ward'
linkage_matrix = linkage(normalized_X, linkage_type)

plt.figure(figsize=(22,18))
dendrogram(linkage_matrix)
plt.show()

"""
Visualization with Seaborn's clustermap
The seaborn plotting library for python can plot a clustermap, which is a detailed
dendrogram which also visualizes the dataset in more detail. It conducts the clustering as
well, so we only need to pass it the dataset and the linkage type we want, and it will use
scipy internally to conduct the clustering
"""
#Expand figsize to a value like (18, 50) if you want the sample labels to be readable
#Draw back is that you'll need more scrolling to observe the dendrogram

sns.clustermap(normalized_X, figsize=(18,50), method=linkage_type, cmap='viridis')



















