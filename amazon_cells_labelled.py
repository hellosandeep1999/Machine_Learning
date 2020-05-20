# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:07:17 2020

@author: user
"""






"""
Q1. Code Challegene (NLP)
Dataset: amazon_cells_labelled.txt


The Data has sentences from Amazon Reviews

Each line in Data Set is tagged positive or negative

Create a Machine learning model using Natural Language Processing that can 
predict wheter a given review about the product is positive or negative

"""

import pandas as pd

# Importing the dataset
# Ignore double qoutes, use 3 
dataset = pd.read_csv('amazon_cells_labelled.txt', delimiter = '\t')

dataset.columns = ["sms","label"]

import nltk
# download the latest list of stopwords from Standford Server 
#nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

import re

corpus = []
 
for i in range(0, 999):
    sms = re.sub('[^a-zA-Z]', ' ', dataset['sms'][i])
    sms = sms.lower()
    sms = sms.split()
    sms = [word for word in sms if not word in set(stopwords.words('english'))]
    
    ps = PorterStemmer()
    sms = [ps.stem(word) for word in sms]
    
    sms = ' '.join(sms)
    
    corpus.append(sms)

print(corpus)
print(len(corpus))




from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)

# it is known as sparse matrix of the features ND Array
features = cv.fit_transform(corpus).toarray() # 2000 columns
labels = dataset.iloc[:, 1].values

print(features.shape)
print(labels.shape)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size = 0.20, random_state = 0)



#applying knn on this text dataset
# Fitting Knn to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()      
classifier.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = classifier.predict(features_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(labels_test, labels_pred)
print(cm_knn)            #0.72
# for better NLP results we need lot of data





-----------------------------------------------------------------------






# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = classifier.predict(features_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(labels_test, labels_pred)
print(cm_nb)       #0.72

#it means Naive bayes and K nearest Neighbors have same solution
















