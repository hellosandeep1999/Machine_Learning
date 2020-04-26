# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:59:37 2020

@author: user
"""

import pandas as pd
dataset = pd.read_csv("cars.csv")

#we need to check any column is null or not if null than fill it
dataset.isnull().any(axis=0)

#only mileage column is null so fill the mileage column by mean()
dataset["Mileage"] = dataset["Mileage"].fillna(dataset["Mileage"].mean())

#again check the null column is not than go on next step
dataset.isnull().any(axis=0)

#distributed features and labels
features = dataset.iloc[:,1:].values
labels = dataset.iloc[:,0].values


#from here first we need to label encoding for all categorical column

#for "make" column
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
features[:, 1] = labelencoder.fit_transform(features[:, 1])

#for "model" column
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
features[:, 2] = labelencoder.fit_transform(features[:, 2])

#for "trim" column
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
features[:, 3] = labelencoder.fit_transform(features[:, 3])

#for "type" column
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
features[:, 4] = labelencoder.fit_transform(features[:, 4])



#now we have completed only label enciding
#from here we starting the onehotencoding

#from here you need to check again and again features from variable explorer

#for column = 0
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [1])
features = onehotencoder.fit_transform(features).toarray()
features = features[:, 1:]

#for column = 5
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [5])
features = onehotencoder.fit_transform(features).toarray()
features = features[:, 1:]

#for column = 32
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [32])
features = onehotencoder.fit_transform(features).toarray()
features = features[:, 1:]

#for column = 73
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [73])
features = onehotencoder.fit_transform(features).toarray()
features = features[:, 1:]


#Making of csv file
df = pd.DataFrame(features)
df.to_csv("cars_only_Features.csv")



