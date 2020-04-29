# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:35:41 2020

@author: user
"""

"""

Code Challenges 02: (House Data) kc_house_data.csv

This is kings house society data.
In particular, we will: 
• Use Linear Regression and see the results
• Use Lasso (L1) and see the resuls
• Use Ridge and see the score
"""


--------------------------------------------------------------------------------------------

#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data Analysis
data = pd.read_csv("Kc_house_data.csv")
data.head()
data.isnull().any(axis=0)
data['sqft_above'] = data['sqft_above'].fillna(data['sqft_above'].mean())
--------------------------------------------------------




"""
• Use Linear Regression and see the results
"""




#Lets build the models now
#Multiple Linear Regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

features = data.drop(['id','date','price','zipcode'], axis=1).values #drop the target to get the features
labels = data['price'].values.reshape(-1,1) #choose the target

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features = sc.fit_transform(features)


lin_reg = LinearRegression()

lin_reg.fit(features,labels)

print(lin_reg.score(features,labels))  #0.6954126317668505

MSEs = cross_val_score(lin_reg, features, labels, scoring='neg_mean_squared_error', cv=5)

mean_MSE = np.mean(MSEs)

print(mean_MSE)

----------------------------------------------------------------

#Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

#alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(features, labels)

ridge_regressor.best_params_      #{'alpha': 20}
ridge_regressor.best_score_       #-41690440333.26298


#Lasso
from sklearn.linear_model import Lasso

lasso = Lasso()
"""
For ridge regression, we introduce GridSearchCV. 
This will allow us to automatically perform 5-fold cross-validation with a range of different regularization parameters in order to find the optimal value of alpha.
"""

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 5)

lasso_regressor.fit(features, labels)

lasso_regressor.best_params_       #{'alpha': 20}
lasso_regressor.best_score_       #-41691943167.15246












