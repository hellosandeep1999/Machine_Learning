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
print('Train Score: ', regressor.score(features_train, labels_train))  
print('Test Score: ', regressor.score(features_test, labels_test))

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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Salary_Classification.csv")

features = dataset.iloc[:,:-1].values
labels = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

features[:, 0] = labelencoder.fit_transform(features[:, 0])

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
features = onehotencoder.fit_transform(features).toarray()

features = features[:, 1:]

import statsmodels.api as sm
features = sm.add_constant(features)

#checking for first time
features_opt = features[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()


#remove x2 and checking for second time
features_opt = features[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()


#remove x3 and checking for third time
features_opt = features[:, [0, 1, 3, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()


#remove x4 and checking for fourth time
features_opt = features[:, [0,3, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()


#remove x4 and checking for fourth time
features_opt = features[:, [0,5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()


#now we get the our column
dataset = dataset.iloc[:,3:]
x_BE= dataset.iloc[:, :-1].values  
y_BE= dataset.iloc[:, -1].values  
  
  
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_BE, y_BE, test_size= 0.2, random_state=0)  
  
#Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_BE_train,y_BE_train)
regressor.fit(nm.array(x_BE_train).reshape(-1,1), y_BE_train)  
  
#Predicting the Test set result;  
y_pred= regressor.predict(x_BE_test)  

dataframe = pd.DataFrame({"Actual":y_BE_test,"Predicted":y_pred})
  
#Cheking the score  
print('Train Score: ', regressor.score(x_BE_train, y_BE_train))  
print('Test Score: ', regressor.score(x_BE_test, y_BE_test))  

====================================================================================================


# Polinomial Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Claims_Paid.csv')

features = dataset.iloc[:, 0:1].values
labels = dataset.iloc[:, 1].values

# Fitting Linear Regression to the dataset
# We can avoid splitting since the dataset is too small

from sklearn.linear_model import LinearRegression
lin_reg_1 = LinearRegression()
lin_reg_1.fit(features, labels)

print (lin_reg_1.predict([[1981]]))

plt.scatter(features, labels, color = 'red')   #by simple linear regression
plt.plot(features, lin_reg_1.predict(features), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Year')
plt.ylabel('Claims Paid')
plt.show()


from sklearn.preprocessing import PolynomialFeatures     #by poly nomial regression
poly_object = PolynomialFeatures(degree = 6)

features_poly = poly_object.fit_transform(features)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(features_poly, labels)

print ("Predicting result with Polynomial Regression")
print (lin_reg_2.predict(poly_object.transform([[1981]])))   #predict the value by polinomial


plt.scatter(features, labels, color = 'red')
plt.plot(features, lin_reg_2.predict(poly_object.fit_transform(features)), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Year')
plt.ylabel('Claims Paid')
plt.show()


======================================================================================================


+++++++++++++++++++++++++

Logistic Regrssion 



+++++++++++++++++++++++++++++++


#result(pass or fail) according to study of hours

#BY linear regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

HOURS = [0.50,0.75,1.00,1.25,1.50,1.75,2.00,2.25,2.50,2.75,3.00,3.25,3.50,3.75,4.00,4.25,4.50,4.75,5.00,5.50]
PASS = [0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1]

plt.scatter(HOURS,PASS)


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression() 

# Converted the data into NDArray
regressor.fit(np.array(HOURS).reshape(-1,1), np.array(PASS).reshape(-1,1)) 

plt.scatter(HOURS, PASS, color = 'red')
plt.plot(HOURS, regressor.predict(np.array(HOURS).reshape(-1,1)), color = 'blue')
plt.title('Study Hours and Exam Score')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score: Marks')
plt.show()

#actually it is giving most waste results 

===========================================================================

#heart disease problem

import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

heart = pd.read_csv('Heart_Disease.csv', sep=',',header=0)  

#there is not any categorical data
heart.head()
heart.sample(5)

#checking for null value
heart.isnull().any(axis=0)

labels = heart.iloc[:,9].values 
features = heart.iloc[:,:9].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.25, random_state = 0)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)

# He has already calculated the mean and sd, so we only need to transform
features_test = sc.transform(features_test)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(features_train, labels_train)

probability = classifier.predict_proba(features_test)
print(probability)


# Predicting the class labels ( 0 or 1 )
labels_pred = classifier.predict(features_test)

# Comparing the predicted and actual values
my_frame= pd.DataFrame(labels_pred, labels_test)
print(my_frame)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)  #(68+20)/116 == 0.7586206896551724 

==============================================================================================

++++++++++++++++++++++++++++++++++++++

#KNN Algorithm

+++++++++++++++++++++++++++++++++++++

#Caesarian Data


import sklearn as sk  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Last column marks whether it was caesarian or not ( 1 or 0 )
df = pd.read_csv('caesarian.csv')  

labels = df.iloc[:,5].values 
features = df.iloc[:,:-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.25, random_state = 41)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)


# Fitting Logistic Regression to the Training set
from sklearn.neighbors import KNeighborsClassifier

# When p = 1, this is equivalent to using manhattan_distance (l1), 
# and euclidean_distance (l2) for p = 2
classifier = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski',p = 2) 

classifier.fit(features_train, labels_train)

labels_pred = classifier.predict(features_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)

print(cm)    #14/20  ==  0.7

========================================================================================================


#social networking adds


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# Last column, whether you have clicked the Ad or no
dataset = pd.read_csv('Social_Network_Ads.csv')
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
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
print(cm)     #81/100   ==   0.81    A very good result



# qus was if sandeep age is 20 and salary is 50000 than he will click on add or not

labels_pred = classifier.predict([[20,50000]])   #array([1], dtype=int64)

#yes he click on add

=====================================================================================================

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

========================================================================================================


++++++++++++++++++++++++++++++++++++

#Decision Tree


++++++++++++++++++++++++++++++++++++


#bill authentication


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
dataset = pd.read_csv("bill_authentication.csv")  

dataset.head()
pd.set_option('display.max_columns', None)
dataset.sample(100)


# Finding missing data
dataset.isnull().any(axis=0)

features = dataset.drop('Class', axis=1)
print(features)
print(features.shape)

  
labels = dataset['Class']  
print(labels)
print(labels.shape)


# Train and test split
from sklearn.model_selection import train_test_split  
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20)  

from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(features_train, labels_train)


labels_pred = classifier.predict(features_test) 

# Comparing the predicted and actual values
my_frame= pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})
print(my_frame)


# Evaluating score
# For classification tasks some commonly used metrics are confusion matrix, 
# precision, recall, and F1 score.
from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(labels_test, labels_pred)
print(cm)  

# Model Score = 98.90 times out of 100 model prediction was RIGHT
print((cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]))


#Evaluate the algo
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(labels_test,labels_pred))  
print(classification_report(labels_test,labels_pred))  
print(accuracy_score(labels_test, labels_pred))

==============================================================================================


#petrol_consumption


import pandas as pd
import numpy as np

# This is  a regression problem
dataset = pd.read_csv('petrol_consumption.csv')  

#data analysis
dataset.shape


# Checking for Categorical Data
dataset.head()
pd.set_option('display.max_columns', None)
dataset.sample(10)


# Finding missing data
dataset.isnull().any(axis=0)


features = dataset.drop('Petrol_Consumption', axis=1)  
labels = dataset['Petrol_Consumption'] 

from sklearn.model_selection import train_test_split  
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0) 

from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor()  
regressor.fit(features_train, labels_train)  

labels_pred = regressor.predict(features_test)

df=pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})  
print(df)  


#Evaluating the algorithm
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(labels_test, labels_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(labels_test, labels_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labels_test, labels_pred)))  
print (np.mean(labels))




===============================================================================================

#past hires

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt

dataset = pd.read_csv("PastHires.csv")  

#data analysis
dataset.shape

# Checking for Categorical Data
# Best part is that DT and RF works on the categorical data also
# We do not need to perform the Label encoding for it, Algo does it internally 

dataset.head()
pd.set_option('display.max_columns', None)
dataset.sample(10)


# Finding missing data
dataset.isnull().any(axis=0)


# Preparing the dataset
# This technique of dropping can be used when the label is in between features
features = dataset.drop('Hired', axis=1)
features = features.values

# Label Encoding for Features



# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

#For Column --->  Employed?
features[:, 1] = labelencoder.fit_transform(features[:, 1])

#For Column --->  Level of Education
features[:, 3] = labelencoder.fit_transform(features[:, 3])

#For Column --->  Top-tier school
features[:, 4] = labelencoder.fit_transform(features[:, 4])

#For Column --->  Internet
features[:, 5] = labelencoder.fit_transform(features[:, 5])


#we need to onehotencoding for this column
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [3])
features = onehotencoder.fit_transform(features).toarray()
features = features[:, 1:]




# Label Encoding for Features
labels = dataset['Hired']  
labels = labels.values

#For Column --->  Hired
labels = labelencoder.fit_transform(labels)



# Train and test split
from sklearn.model_selection import train_test_split  
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.20)  

# Training and making predictions 
# We need to be careful in using DecissionTreeClassifier or DecissionTreeRegressor
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(features_train, labels_train)


labels_pred = classifier.predict(features_test) 

# Comparing the predicted and actual values
my_frame= pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})
print(my_frame)


# Evaluating score
# For classification tasks some commonly used metrics are confusion matrix, 
# precision, recall, and F1 score.
from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(labels_test, labels_pred)
print(cm)  

# Model Score = 100% times out of 100 model prediction was RIGHT
print((cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]))


#Evaluate the algo
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(labels_test,labels_pred))  
print(classification_report(labels_test,labels_pred))  
print(accuracy_score(labels_test, labels_pred))



=============================================================================================









 

"""
Build a classification tree model evaluation if an adolescent gets expelled 
or not from school based on their Gender and violent behavior.
Use random forest in relation to regular smokers as a target and explanatory 
variable specifically with Hispanic, White, Black, Native American and Asian.

"""






import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 

# This is  a regression problem
dataset = pd.read_csv('tree_addhealth.csv')  

#data analysis
dataset.shape


# Checking for Categorical Data
dataset.head()
pd.set_option('display.max_columns', None)
dataset.sample(10)


# Finding missing data
dataset.isnull().any(axis=0)

age_mean = dataset["age"].mean()
dataset["age"] = dataset["age"].fillna(age_mean)

dataset.columns


features = dataset.loc[:,['BIO_SEX', 'HISPANIC', 'WHITE', 'BLACK', 'NAMERICAN', 'ASIAN', 'age']].values  
labels = dataset['EXPEL1'].fillna(0)


# Train and test split
from sklearn.model_selection import train_test_split  
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.20)  

# Training and making predictions 
from sklearn.ensemble import RandomForestClassifier


classifier = RandomForestClassifier(n_estimators=20, random_state=0)  
classifier.fit(features_train, labels_train)  

labels_pred = classifier.predict(features_test)

# Comparing the predicted and actual values
my_frame= pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})
print(my_frame)


# Evaluating score
# For classification tasks some commonly used metrics are confusion matrix, 
# precision, recall, and F1 score.
from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(labels_test, labels_pred)
print(cm)  



=======================================================================================================


++++++++++++++++++++++++++++++++++

Random Forest Algorithm

+++++++++++++++++++++++++++++++++


import pandas as pd

dataset = pd.read_csv("bill_authentication.csv")  

dataset.isnull().any(axis=0)

features = dataset.iloc[:, 0:4].values  
labels = dataset.iloc[:, 4].values 

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.2, random_state=0)  

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
features_train = sc.fit_transform(features_train)  
features_test = sc.transform(features_test) 


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=20, random_state=0)  
classifier.fit(features_train, labels_train)  


labels_pred = classifier.predict(features_test) 

# Comparing the predicted and actual values
my_frame= pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})
print(my_frame)


from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(labels_test, labels_pred)
print(cm)  

# Model Score = 98.90 times out of 100 model prediction was RIGHT
print( (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]))


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(labels_test,labels_pred))  
print(classification_report(labels_test,labels_pred))  
print(accuracy_score(labels_test, labels_pred))

==================================================================================================



"""
Problem Definition
The problem here is to predict the gas consumption (in millions of gallons) 
in 48 of the US states based on petrol tax (in cents), per capita income 
(dollars), paved highways (in miles) and the proportion of population with 
the driving license.
"""

#Import libraries
import pandas as pd  
import numpy as np  
dataset = pd.read_csv('petrol_consumption.csv') 

features = dataset.iloc[:, 0:4].values  
labels = dataset.iloc[:, 4].values  

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.2, random_state=0) 


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
features_train = sc.fit_transform(features_train)  
features_test = sc.transform(features_test)  

#train the model
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=25, random_state=0)  
regressor.fit(features_train, labels_train)                       

labels_pred = regressor.predict(features_test)

df=pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})  
print(df) 


#Evaluating the algorithm
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(labels_test, labels_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(labels_test, labels_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labels_test, labels_pred)))  
print (np.mean(labels))


-------------------------------------------

#Change the number of estimators
regressor = RandomForestRegressor(n_estimators=300, random_state=0)  
regressor.fit(features_train, labels_train)  


labels_pred = regressor.predict(features_test)

df=pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})  
print(df) 

print('Mean Absolute Error:', metrics.mean_absolute_error(labels_test, labels_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(labels_test, labels_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labels_test, labels_pred))) 



==================================================================================================


+++++++++++++++++++++++++++++++++++++++++++++++

Model performance metric

+++++++++++++++++++++++++++++++++++++++++++++++


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("student_scores.csv")

dataset.shape
dataset.ndim
dataset.head()
dataset.describe()
dataset.info()
dataset.dtypes

plt.boxplot(dataset.values)

#plt.scatter(dataset["Hours"],dataset["Scores"]) 

dataset.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

#prepare the data to train the model
features = dataset.iloc[:, :-1].values  
labels = dataset.iloc[:, 1].values 

from sklearn.model_selection import train_test_split  
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)  

#train the algo
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(features_train, labels_train)  

#To see the value of the intercept and slop calculated by the linear regression algorithm for our dataset, execute the following code.
print(regressor.intercept_)  
print (regressor.coef_)

"""
Defination of Coefficient
This means that for every one unit of change in hours studied ( x axis), 
the change in the score(y axis)  is about 9.91%. 
"""


labels_pred = regressor.predict(features_test) 
df = pd.DataFrame({'Actual': labels_test, 'Predicted': labels_pred})  
print ( df )


#Visualize the best fit line
import matplotlib.pyplot as plt

# Visualising the Test set results
plt.scatter(features_test, labels_test, color = 'red')
plt.plot(features_train, regressor.predict(features_train), color = 'blue')
plt.title('Study Hours and Exam Score')
plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.show()



===========================================================================================
# Logistic Regression ( Classification)


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = classifier.predict(features_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)


from sklearn.metrics import precision_score
 
# Take turns considering the positive class either 0 or 1
print (precision_score(labels_test, labels_pred, pos_label=0)  )
print (precision_score(labels_test, labels_pred, pos_label=1)  )

===============================================================================================


++++++++++++++++++++++++++

Regularization

++++++++++++++++++++++++++++


#Bosten dataset


import numpy as np 
import pandas as pd 
import sklearn
import seaborn as sns
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn import datasets

boston = datasets.load_boston()

print(boston.DESCR)

#inforation about boston dataset
boston.keys() 
boston.data  # ( features )
boston.data.shape
boston.feature_names
boston.target 

boston_df = pd.DataFrame(boston.data,columns=boston.feature_names)
boston_df.head()

#add dependent variables
boston_df['House_Price']= boston.target
boston_df.head ()
boston_df.describe()

features = boston_df.drop('House_Price',axis = 1)
labels = boston_df['House_Price']
features.head()
labels.head()


#Create train and test data with 70o/o and 30°/o split
features_train, features_test, labels_train,labels_test  =  train_test_split(features, labels, test_size=0.3, random_state=1)

features_train.shape

features_test.shape

labels_train.shape
labels_test.shape

from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso 
from sklearn.linear_model import Ridge  # RidgeClassier is also there
from sklearn.linear_model import ElasticNet
lm = LinearRegression ()
lm_lasso = Lasso() 
lm_ridge =  Ridge() 
lm_elastic = ElasticNet() 


#Fit a model on the train data
lm.fit(features_train, labels_train)
lm_lasso.fit(features_train, labels_train)
lm_ridge.fit(features_train, labels_train)
lm_elastic.fit(features_train, labels_train)

plt.figure (figsize= (15,10))
ft_importances_lm = pd.Series(lm.coef_, index= features.columns)
ft_importances_lm .plot(kind = 'barh')
plt.show()


print ("RSquare Value for Simple Regresssion TEST data is-") 
print (np.round (lm .score(features_test,labels_test)*100,2))

print ("RSquare Value for Lasso Regresssion TEST data is-")
print (np.round (lm_lasso.score(features_test,labels_test)*100,2))

print ("RSquare Value for Ridge Regresssion TEST data is-")
print (np.round (lm_ridge.score(features_test,labels_test)*100,2))

print ("RSquare Value for Elastic Net Regresssion TEST data is-")
print (np.round (lm_elastic.score(features_test,labels_test)*100,2))

#Predict on test and training data

predict_test_lm =	lm.predict(features_test ) 
predict_test_lasso = lm_lasso.predict (features_test) 
predict_test_ridge = lm_ridge.predict (features_test)
predict_test_elastic = lm_elastic.predict(features_test)

#Print the Loss Funtion - MSE & MAE

import numpy as np
from sklearn import metrics
print ("Simple Regression Mean Square Error (MSE) for TEST data is") 
print (np.round (metrics .mean_squared_error(labels_test, predict_test_lm),2) )

print ("Lasso Regression Mean Square Error (MSE) for TEST data is") 
print (np.round (metrics .mean_squared_error(labels_test, predict_test_lasso),2))

print ("Ridge Regression Mean Square Error (MSE) for TEST data is") 
print (np.round (metrics .mean_squared_error(labels_test, predict_test_ridge),2))

print ("ElasticNet Mean Square Error (MSE) for TEST data is")
print (np.round (metrics .mean_squared_error(labels_test, predict_test_elastic),2))

==========================================================================================



#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data Analysis
data = pd.read_csv("Advertising.csv")
data.head()

#Drop the first column

data.drop(['Unnamed: 0'], axis=1, inplace=True)

print (data.head())

print (data.columns)

#lets plot few visuals
def scatter_plot(feature, target):
    plt.scatter(data[feature], data[target], c='black')
    plt.xlabel("Money spent on {} ads ($)".format(feature))
    plt.ylabel("Sales ($k)")
    plt.show()

scatter_plot('TV', 'sales')
scatter_plot('radio', 'sales')
scatter_plot('newspaper', 'sales')

#Lets build the models now
#Multiple Linear Regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

features = data.drop(['sales'], axis=1) #drop the target to get the features
labels = data['sales'].values.reshape(-1,1) #choose the target

lin_reg = LinearRegression()

MSEs = cross_val_score(lin_reg, features, labels, scoring='neg_mean_squared_error', cv=5)

mean_MSE = np.mean(MSEs)

print(mean_MSE)


#Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

#alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(features, labels)

ridge_regressor.best_params_
ridge_regressor.best_score_


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

lasso_regressor.best_params_
lasso_regressor.best_score_


=======================================================================================







































































