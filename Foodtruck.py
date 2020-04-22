# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:53:22 2020

@author: user
"""

"""
Code Challenge
  Name: 
    Food Truck Profit Prediction Tool
  Filename: 
    Foodtruck.py
  Dataset:
    Foodtruck.csv
  Problem Statement:
    Suppose you are the CEO of a restaurant franchise and are considering 
    different cities for opening a new outlet. 
    
    The chain already has food-trucks in various cities and you have data for profits 
    and populations from the cities. 
    
    You would like to use this data to help you select which city to expand to next.
    
    Perform Simple Linear regression to predict the profit based on the 
    population observed and visualize the result.
    
    Based on the above trained results, what will be your estimated profit, 
    
    If you set up your outlet in Jaipur? 
    (Current population in Jaipur is 3.073 million)
        
  Hint: 
    You will implement linear regression to predict the profits for a 
    food chain company.
    Foodtruck.csv contains the dataset for our linear regression problem. 
    The first column is the population of a city and the second column is the 
    profit of a food truck in that city. 
    A negative value for profit indicates a loss.
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Foodtruck.csv")

#data.isnull().any(axis=0)

#plt.boxplot(data.values)

#x = list(data["Population"])
#y = list(data["Profit"])

#plt.scatter(x,y)

features = data.iloc[:,:-1].values
labels = data.iloc[:,1:].values

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(features,labels)
plt.scatter(features,labels,color="red")

plt.plot(features,reg.predict(features),color="Blue")

z = reg.predict([[3.073]])
if z < 0:
    print("Loss - ", z)
else:
    print("Profit - ", z)
    
"""
You would like to use this data to help you select which city to expand to next.

"""
 
data.loc[data["Profit"] == data["Profit"].max()]

