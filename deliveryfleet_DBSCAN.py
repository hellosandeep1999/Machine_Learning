# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:29:40 2020

@author: user
"""

"""
Q1. (Create a program that fulfills the following specification.)
deliveryfleet.csv


Import deliveryfleet.csv file

Here we need Two driver features: mean distance driven per day (Distance_feature) 
and the mean percentage of time a driver was >5 mph over the speed limit (speeding_feature).

Perform K-means clustering to distinguish urban drivers and rural drivers.
Perform K-means clustering again to further distinguish speeding drivers 
from those who follow speed limits, in addition to the rural vs. urban division.
Label accordingly for the 4 groups.

"""

------------------------------------------------------------------------------------------





"""

Here we need Two driver features: mean distance driven per day (Distance_feature) 
and the mean percentage of time a driver was >5 mph over the speed limit (speeding_feature).

"""


import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


dataset = pd.read_csv('deliveryfleet.csv')

features = dataset.iloc[:, [1, 2]].values

plt.scatter(features[:,0], features[:,1])
plt.show()

db = DBSCAN(eps=3, min_samples=10)

model = db.fit(features)

labels = model.labels_ 

sample_cores = np.zeros_like(labels, dtype=bool)
sample_cores[db.core_sample_indices_] = True




n_clusters = len(set(labels))- (1 if -1 in labels else 0)

print(metrics.silhouette_score(features,labels))   #0.4945869610378622



--------------------------------------------------------------------------------------------









