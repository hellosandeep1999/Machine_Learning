# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 00:03:24 2020

@author: user
"""

#simple Linear Regrassion

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing our dataset
df = pd.read_csv("student_scores.csv")


#now differentiate the varaibles into dependent and independent variables
features = df.iloc[:,:1].values
labels = df.iloc[:,-1].values


#next step is divide the features and label into train and test data
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2,random_state = 0)



#here we will do ready to the our model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train,labels_train)

labels_pred =  regressor.predict(features_test)

plt.scatter(features_train, labels_train, color="green")   
plt.plot(features_train, regressor.predict(features_train), color="red")    
plt.title("Salary vs Experience (Training Dataset)")  
plt.xlabel("Years of Experience")  
plt.ylabel("Salary(In Rupees)")  
plt.show() 

myframe = pd.DataFrame({"Actual" : labels_test,"Predicted" : labels_pred})
print(myframe)

#here we have completed our Simple Linear Regression


==========================================================================================================



#Bahubali2vsDangal



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

x = int(input("Enter a number:  "))
dataset = pd.read_csv("Bahubali2_vs_Dangal.csv")

features = dataset.iloc[:,:-2]
labels1 = dataset.iloc[:,1:-1]
labels2 = dataset.iloc[:,2:]
from sklearn.linear_model import LinearRegression

reg1 = LinearRegression()
reg2 = LinearRegression()

reg1.fit(features,labels1)
reg2.fit(features,labels2)

plt.scatter(features,labels1,color="red")
plt.scatter(features,labels2,color="red")

plt.plot(features,reg.predict(features),color="Blue")
plt.plot(features,reg.predict(features),color="Blue")

a = reg1.predict([[x]])
b = reg2.predict([[x]])

if a > b:
    print("Bahubali2 collection of",x,"th day will be maximum = ",a)
else:
    print("Dangal collection of",x,"th day will be maximum = ",b)



=========================================================================================================


#Foodtruck

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





========================================================================================================










