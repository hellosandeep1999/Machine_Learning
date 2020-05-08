# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:31:02 2020

@author: user
"""

"""

Q1.bank_note.csv

Program Specification

Suppose you are the manager of a bank and you have the problem of discriminating 
between genuine and counterfeit banknotes. 
You are measuring several distances on the banknote and the width and height of it.

Measuring these values of about 100 genuine and 100 counterfeit banknotes, 
Use the data set to set up a logical regression and is capable of discriminating 
between genuine and counterfeit money classification. (Import banknotes.csv)

(this data set contains data on Swiss francs currency; it has been obtained courtesy of H. Riedwyl )

Check the accuracy of your model using confusion matrix.

Then use k-fold cross validation to find actual mean accuracy of your model.


"""


----------------------------------------------------------------------------------------------







"""

Check the accuracy of your model using confusion matrix.

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('banknotes.csv')
features = dataset.iloc[:, [1,2,3,4,5]].values
labels = dataset.iloc[:, 6].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

# Fitting Knn to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = classifier.predict(features_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)

"""
[[27  0]
 [ 0 23]]
"""
print( (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]))  #1.0 it means it is best prediction



---------------------------------------------------------------------------------------------------



  
"""
Then use k-fold cross validation to find actual mean accuracy of your model.

"""


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = features_train, y = labels_train, cv = 10)
print ("accuracies is ", accuracies)
print ("mean accuracy is",accuracies.mean()) 



#mean accuracy is 0.940654761904762   --->  Very Good model and Prediction



















