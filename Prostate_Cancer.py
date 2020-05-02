# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:42:48 2020

@author: user
"""
"""
Code Challenge 01: (Prostate Dataset) Prostate_Cancer.csv

This is the Prostate Cancer dataset. Perform the train test split before you apply the model.


"""

------------------------------------------------------------------------------------------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Prostate_Cancer.csv")  #from csv
dataset.isnull().any(axis=0)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
dataset.iloc[:, 1] = labelencoder.fit_transform(dataset.iloc[:, 1])

#prepare the data to train the model
features = dataset.iloc[:, [2,3,4,5,6,8]].values  
labels = dataset.iloc[:, 1].values 


from sklearn.model_selection import train_test_split  
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)  

#train the algo
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(features_train, labels_train)  

labels_pred = regressor.predict(features_test)
data = pd.DataFrame({"Actual":labels_test,"Predicted":labels_pred})

print('Train Score: ', regressor.score(features_train, labels_train))  #0.39410555070636477
print('Test Score: ', regressor.score(features_test, labels_test))     #0.22746767101186915



#very BAD model


