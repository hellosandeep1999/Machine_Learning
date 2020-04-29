# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:50:28 2020

@author: user
"""

"""

Code Challenge 02:(pros.dat.csv)

Load the dataset from given link:
pd.read_csv("http://www.stat.cmu.edu/~ryantibs/statcomp/data/pros.dat", delimiter =' ')

(a) Can we predict lpsa from the other variables?
      (1) Train the unregularized model (linear regressor) and calculate the mean squared error.
      (2) Apply a regularized model now - Ridge regression and lasso as well and check the mean squared error.

(b)Can we predict whether lpsa is high or low, from other variables?
"""



-------------------------------------------------------------------------------------------

"""
(a) Can we predict lpsa from the other variables?
 (1) Train the unregularized model (linear regressor) and calculate the mean squared error.
 
 """



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv("http://www.stat.cmu.edu/~ryantibs/statcomp/data/pros.dat",delimiter = " ")
dataset.isnull().any(axis=0)

features = dataset.iloc[:, :-1].values  
labels = dataset.iloc[:, -1].values 
from sklearn.model_selection import train_test_split  
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)  

#train the algo
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(features_train, labels_train)  

labels_pred = regressor.predict(features_test)
data = pd.DataFrame({"Actual":labels_test,"Predicted":labels_pred})


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(labels_test, labels_pred))  
print('Mean Squared Error:', 
      metrics.mean_squared_error(labels_test, labels_pred))  
print('Root Mean Squared Error:', 
      np.sqrt(metrics.mean_squared_error(labels_test, labels_pred))) 



----------------------------------------------------------------------------------------

"""
(b) Can we predict whether lpsa is high or low, from other variables?

"""

labels_pred = regressor.predict(features_test)
data = pd.DataFrame({"Actual":labels_test,"Predicted":labels_pred})

#yes we can predict lpsa low or high but we need some of data like features test 










