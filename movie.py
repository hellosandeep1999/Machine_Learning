# -*- coding: utf-8 -*-
"""
Created on Wed May 13 00:13:56 2020

@author: user
"""

"""
Q3 movie.csv 

Program Specification

Import movie.csv file

There are two categories: Pos (reviews that express a positive or favorable sentiment) 
and Neg (reviews that express a negative or unfavorable sentiment). 
For this assignment, we will assume that all reviews are either positive or negative; 
there are no neutral reviews.

Perform sentiment analysis on the text reviews to determine whether its positive 
or negative and build confusion matrix to determine the accuracy.

"""




import pandas as pd

dataset = pd.read_csv("movie.csv")

import nltk
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer  #For Stemming use the PorterStemmer class object 

import re

corpus = []
 
for i in range(0, 2000):
    text = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    text = text.lower()
    text = text.split()
    text = [word for word in text if not word in set(stopwords.words('english'))]
    
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]
    
    text = ' '.join(text)
    
    corpus.append(text)

print(corpus)
print(len(corpus))



from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()






#Now we will check the accuracy of our model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)

# it is known as sparse matrix of the features ND Array
features = cv.fit_transform(corpus).toarray() # 1500 columns
labels = labelencoder.fit_transform(dataset.loc[:, ['class']])

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
print(cm_knn)            #0.605
# for better NLP results we need lot of data


#we get the accuracy  ---->  60.5%

--------------------------------------------------------------------------------------------





# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = classifier.predict(features_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(labels_test, labels_pred)
print(cm_nb)       #0.725

#we get the accuracy  ---->  72.5%


----------------------------------------------------------




"""
Final Result :-

So we can say that here Naive Bayes algorithm is working better and best accuracy giving 72.5%
it is a good accuracy for sentimental data.

"""















