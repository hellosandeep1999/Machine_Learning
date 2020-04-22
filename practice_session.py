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







++++++++++++++++++++++++++++++++++++++++++++++

#MultilinearRegression Strars from here

++++++++++++++++++++++++++++++++++++++++++++






#Salary_Classification


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Classification.csv')

features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values


dataset.isnull().any(axis=0)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder

# Create objct of LabelENcoder
labelencoder = LabelEncoder()

# Fit and Use the operation Transform
features[:, 0] = labelencoder.fit_transform(features[:, 0])


from sklearn.preprocessing import OneHotEncoder


onehotencoder = OneHotEncoder(categorical_features = [0])

# Convert to NDArray format
features = onehotencoder.fit_transform(features).toarray()

features = features[:,1:]

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
# Whether we have Univariate or Multivariate, class is LinearRegression

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train, labels_train)

print(regressor.intercept_)   
print (regressor.coef_) 

Pred = regressor.predict(features_test)

print (pd.DataFrame(zip(Pred, labels_test)))

# Prediction for a new values for a person in 'Development', hours worked 1150,
# 3 certificates , 4yrs of experience. What would be his salary ??




x = ['Development',1150,3,4]

x = np.array(x)
x = x.reshape(1,4)
regressor.predict(x)
#this will show you error


x = [1,0,0,1150,3,4]
x = np.array(x)
x = x.reshape(1,4)
regressor.predict(x) #this also show you the error

x = [0,0,1150,3,4]
x = np.array(x)
x = x.reshape(1,5)
regressor.predict(x)


Score = regressor.score(features_train, labels_train)
Score = regressor.score(features_test, labels_test)









=======================================================================================================







#cars file




import pandas as pd
import numpy as np


# Importing the dataset
dataset = pd.read_csv('cars.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values.reshape(-1,1)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)
print (X_train,X_test,y_train, y_test)


# Write code to save in the csv file

# Combining the features and labels in both train and test data
train_data = np.concatenate([X_train, y_train],axis=1)
test_data = np.concatenate([X_test, y_test], axis=1)

# Fetching all the columns name from the original dataset
head = list(dataset.columns)

# Framing the test and train dataframe
train_df, test_df = pd.DataFrame(), pd.DataFrame()

for var in range(0,12):
    train_df[head[var]] = train_data[:, var]
    test_df[head[var]] = test_data[:, var]

# Creating seperate train and test csv files
train_df.to_csv("cars_train.csv")
test_df.to_csv("cars_test.csv")

# Printing the train and test dataframes
print("train_data:", train_df)
print("test_data:", test_df)



=================================================================================================

#iq size



import pandas as pd
import numpy as np

df = pd.read_csv("iq_size.csv")

print(df.dtypes)

#check the datset have null value or not 
#if it is have null then we need to handle it.

df.isnull().any(axis=0)

#here we need to seprate features and labels

features = df.iloc[:,1:].values
label = df.iloc[:,0].values 

from sklearn.model_selection import train_test_split

features_train,features_test,label_train,label_test = train_test_split(features,label,test_size = 0.2, random_state=0)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(features_train,label_train)

pred = reg.predict(features_test)

print(pd.DataFrame(zip(pred,label_test)))

x = [90,70,150]

x = np.array(x)
x = x.reshape(1,3)
reg.predict(x)



=================================================================================================

#female states


==================================================================================================




++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#Backword Elimination
#Polinomial

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

























