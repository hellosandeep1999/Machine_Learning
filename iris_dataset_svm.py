# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:33:39 2020

@author: user
"""

#iris dataset


import numpy as np 
import pandas as pd 
import sklearn
import seaborn as sns
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()

print(iris.DESCR)

#inforation about boston dataset
iris.keys() 
iris.data  # ( features )
iris.data.shape
iris.feature_names
iris.target 

iris_df = pd.DataFrame(iris.data,columns=iris.feature_names)
iris_df.head()


#add dependent variable
iris_df['flower_species']= iris.target
iris_df.head ()
iris_df.describe()


features = iris_df.drop('flower_species',axis = 1)
labels = iris_df['flower_species']

features_train, features_test, labels_train,labels_test  =  train_test_split(features, labels, test_size=0.3, random_state=1)



from sklearn.svm import SVC
# SVM ( SVC for classification and SVR for Regression )
classifier = SVC(kernel = 'poly', random_state = 0)
classifier.fit(features_train, labels_train)

labels_pred = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)

# Model Score
print( (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]))

score = classifier.score(features_test,labels_test)
print(score)   #1.0


#model is perfect











