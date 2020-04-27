# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:45:17 2020

@author: user
"""

#tree add health


"""

Build a classification tree model evaluating if an adolescent would smoke regularly 
    or not based on: gender, age, (race/ethnicity) Hispanic, White, Black, Native American 
    and Asian, alcohol use, alcohol problems, marijuana use, cocaine use, inhalant use, 
    availability of cigarettes in the home, depression, and self-esteem.
    
    
    
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('tree_addhealth.csv')

dataset = dataset.drop(dataset.index[88])
dataset.isnull().any(axis=0)

dataset["age"] = dataset["age"].fillna(dataset["age"].mean())
dataset["SCHCONN1"] = dataset["SCHCONN1"].fillna(dataset["SCHCONN1"].mean())
dataset["GPA1"] = dataset["GPA1"].fillna(dataset["GPA1"].mean())
dataset["PARPRES"] = dataset["PARPRES"].fillna(dataset["PARPRES"].mean())


features = dataset.iloc[:, [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15]].values
labels = dataset.iloc[:, 7].values

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.25, random_state = 40)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(features_train, labels_train)

#Calculate Class Probabilities
probability = classifier.predict_proba(features_test)

# Predicting the class labels
labels_pred = classifier.predict(features_test)

data = pd.DataFrame({"Actual":labels_test,"Predicted":labels_pred})

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)   # 0.9545454545454546  it is very good prediction
