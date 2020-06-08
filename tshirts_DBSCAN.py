# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:49:43 2020

@author: user
"""

"""

Q2. (Create a program that fulfills the following specification.)
tshirts.csv


T-Shirt Factory:

You own a clothing factory. You know how to make a T-shirt given the height 
and weight of a customer.

You want to standardize the production on three sizes: small, medium, and large. 
How would you figure out the actual size of these 3 types of shirt to better 
fit your customers?

Import the tshirts.csv file and perform Clustering on it to make sense out of 
the data as stated above.


"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

dataset = pd.read_csv('tshirts.csv')

features = dataset.iloc[:, [1, 2]].values

plt.scatter(features[:,0], features[:,1])
plt.show()

db = DBSCAN(eps=5, min_samples=3)

model = db.fit(features)

labels = model.labels_ 


n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)


#sample_cores = np.zeros_like(labels, dtype=bool)
#sample_cores[db.core_sample_indices_] = True



plt.scatter(features[labels== 0,0], features[labels == 0,1],c='red', marker='+' )
plt.scatter(features[labels == 1,0], features[labels == 1,1],c='green', marker='o' )
plt.scatter(features[labels == -1,0],features[labels == -1,1],c='yellow', marker='*' )


print(metrics.silhouette_score(features,labels))  #0.4634948611025891


-------------------------------------------------------------------------------------