# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:00:16 2020

@author: user
"""

"""

Auto_mpg.txt

Here is the dataset about cars. The data concerns city-cycle fuel consumption in miles per gallon (MPG).

    Import the dataset Auto_mpg.txt
    Give the column names as "mpg", "cylinders", "displacement","horsepower","weight",
    "acceleration", "model year", "origin", "car name" respectively
    Display the Car Name with highest miles per gallon value
    Build the Decision Tree and Random Forest models and find out which of the two is more accurate in predicting the MPG value
    
    Find out the MPG value of a 80's model car of origin 3, weighing 2630 kgs with 6 cylinders, having acceleration around 
    22.2 m/s due to it's 100 horsepower engine giving it a displacement of about 215. (Give the prediction from both the models)




"""


# first we need to convert .txt file into .csv file so using file handling we will

 


import pandas as pd
fp = open(r'Auto_mpg.txt','r')
a=[]
b=[]
c=[]
d=[]
e=[]
f=[]
g=[]
h=[]
i=[]
while(True):
    data=fp.readline()
    print(data.split())
    if data!='':
#         a,b,c,d,e,f,g,h,i=data.split()
        a.append(data.split()[0])
        b.append(data.split()[1])
        c.append(data.split()[2])
        d.append(data.split()[3])
        e.append(data.split()[4])
        f.append(data.split()[5])
        g.append(data.split()[6])
        h.append(data.split()[7])
        string = data.split()[8:]
        string = " ".join(string)
        string = string[1:]
        string = string[:-1]
        i.append(string)
        
    else:
        break
# print() 
df=pd.DataFrame(zip(a,b,c,d,e,f,g,h,i))
# pd.to_csv('auto_mpg.csv')
df.to_csv('Auto_mpg.csv', encoding='utf-8')

------------------------------------------------------------




"""

Give the column names as "mpg", "cylinders", "displacement","horsepower","weight",
    "acceleration", "model year", "origin", "car name" respectively
    
"""


#now from here we will solve the problem 

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt

dataset = pd.read_csv("Auto_mpg.csv")
dataset.drop("Unnamed: 0",inplace=True,axis= 1)
dataset.columns = ["mpg", "cylinders", "displacement","horsepower","weight","acceleration", "model year", "origin", "car name"]

dataset["horsepower"] = dataset["horsepower"].replace(to_replace = "?",value = "140.0")

dataset["horsepower"] = dataset.horsepower.astype(float)



dataset.dtypes



dataset.isnull().any(axis=0)




---------------------------------------------------------------------------------------


"""

Display the Car Name with highest miles per gallon value


"""
dataset[dataset["mpg"] == dataset["mpg"].max()]["car name"]





--------------------------------------------------------------------------------------









"""
Build the Decision Tree and Random Forest models and find out which of the two is more accurate 
in predicting the MPG value

"""




dataset.isnull().any(axis=0)


# Preparing the dataset
# This technique of dropping can be used when the label is in between features
features = dataset.iloc[:,[1,2,3,4,5,6,7]].values


  
labels = dataset.iloc[:,0].values


# Train and test split
from sklearn.model_selection import train_test_split  
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.20)  

# Training and making predictions 
# We need to be careful in using DecissionTreeClassifier or DecissionTreeRegressor

#this predoiction from decision tree
from sklearn.tree import DecisionTreeRegressor   
regressor =DecisionTreeRegressor()  

regressor.fit(features_train, labels_train)

labels_pred = regressor.predict(features_test) 


# Comparing the predicted and actual values
my_frame= pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})
print(my_frame)

print('Train Score: ', regressor.score(features_train,labels_train))            # 1.0
print('Test Score: ', regressor.score(features_test,labels_test))               #  0.8571344089295849



"""

Find out the MPG value of a 80's model car of origin 3, weighing 2630 kgs with 6 cylinders, having acceleration around 
    22.2 m/s due to it's 100 horsepower engine giving it a displacement of about 215. (Give the prediction from both the models)


"""#by dicisionTreeregressor
labels_pred = regressor.predict([[6,215,100,2630,22.2,80,3]])   #array([20.2])



----------------------------------------------------------------------------------------------------------------------------------------------------







#And this prection from random forest






from sklearn.ensemble import RandomForestRegressor 
  
 # create regressor object 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
  
# fit the regressor with x and y data 
regressor.fit(features_train, labels_train) 


labels_pred = regressor.predict(features_test) 

# Comparing the predicted and actual values
my_frame= pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})
print(my_frame)

print('Train Score: ', regressor.score(features_train,labels_train))            #  0.9815130997386704
print('Test Score: ', regressor.score(features_test,labels_test))              #    0.872313257337975

"""

Find out the MPG value of a 80's model car of origin 3, weighing 2630 kgs with 6 cylinders, having acceleration around 
    22.2 m/s due to it's 100 horsepower engine giving it a displacement of about 215. (Give the prediction from both the models)


"""
#by RandomForestRegressor 

labels_pred = regressor.predict([[6,215,100,2630,22.2,80,3]])    #array([24.182])




























