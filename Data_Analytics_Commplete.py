# -*- coding: utf-8 -*-
"""
Created on Thu May  7 20:27:08 2020

@author: user
"""


------------------------------------------------------------------------------------------------------


+++++++++++++++++++++++++++++++++++++
"""
Basics of Matplotlib
    Scatter Plot
    Line Plot
    Pie Chart
    Bar Chart
    Histogram
"""


+++++++++++++++++++++++++++++++++++++++++








# Open anatomy_of_matplotlib0
# Open anatomy_of_matplotlib

#%matplotlib
import matplotlib
matplotlib.__version__

import matplotlib.pyplot as plt


x = [1,2,3,4,5,6,7,8,9,10]

y = [1,2,3,4,5,6,7,8,9,10]


# Showing the points on the graph
plt.scatter(x, y)

# Simple Line plot
plt.plot(x, y)


# Setting the title
plt.title("A Line Graph")

# Setting the X Label 
plt.xlabel("X")

# Setting the Y Label
plt.ylabel("Y")

# Displaying the Grid
plt.grid(True)

# Changing the x axes limits of the scale
plt.xlim(0, 10)

# Changing the y axes limits of the scale
plt.ylim(0, 10)

# Or
plt.axis([0, 10, 0, 10])


# Showing the points on the graph
plt.scatter(x, y)

# Simple Line plot
plt.plot(x, y)

plt.savefig("scatter.jpg")

plt.show()


# Changing the color of the line
plt.plot(x, y, color='green') # #000000
plt.plot(x, y, color="#FF0000") # #000000
         
         
# Changing the style of the line
plt.plot(x, y, linestyle='dashed') # solid dashed  dashdot dotted

# Changing the linewidth
plt.plot(x, y, linewidth=5.0) 

# For Plotting Scatter Plot
plt.plot(x, y, 'd', color='red') # o  .  , x  +  v  ^  <  >  s d 


# Summary of all the attributes
plt.plot(x, y,
         linewidth=2.0,
         linestyle='solid',
         color='red',
         alpha=1.0,
         marker='o')



# Scatter Plot with scatter method 
plt.scatter(x, y, marker='.', color='black',label="marker='{0}'".format('.')); # o  .  , x  +  v  ^  <  >  s d 
plt.legend(numpoints=1)


plt.scatter(5, 5, s=180, color='yellowgreen', alpha=0.9)

'''
# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
z = [1,2,3,4,5,6,7,8,9,10]
fig, ax = plt.subplots()
ax.plot(x,y)
ax.plot(x,z)
'''







--------------------------------------------------------------------------------






++++++++++++++++++++++++++++++++++
"""
Pie chart, where the slices will be ordered and plotted counter-clockwise:
"""

++++++++++++++++++++++++++++++++++++++++++++++







labels = ['CSE', 'ECE', 'IT', 'EE']
sizes = [15, 30, 25, 10]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = [0.1, 0, 0, 0]  # explode 1st slice

#plt.pie(sizes, labels=labels, autopct='%.0f%%')

# or

plt.pie(sizes, labels=labels, explode=explode, colors=colors, autopct='%1.2f%%', shadow=True, startangle=90)


plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

        

"""
Plotting a bar chart
"""

import matplotlib.pyplot as plt

performance = [10,8,6,4,2,1]
index = [0,1,2,3,4,5]
plt.bar(index, performance, align='center', alpha=1.0)

objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
plt.xticks(index, objects)
plt.ylabel('Usage')
plt.title('Programming Language Usage')
 
plt.show()

------------------------------------------------------------


++++++++++++++++++++++++++++

# Another Example of Bar Chart
  

+++++++++++++++++++++++++++++++++


import matplotlib.pyplot as plt

 
# 14 categories of movies
label = ['Adventure', 'Action', 'Drama', 'Comedy', \
         'Thriller/Suspense', 'Horror', 'Romantic Comedy', 'Musical', \
         'Documentary', 'Black Comedy', 'Western', 'Concert/Performance', \
         'Multiple Genres', 'Reality']
 
no_movies = [941,854,4595,2125,942,509,548,149,1952,161,64,61,35,5]

index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

plt.bar(index, no_movies)
plt.xlabel('Genre', fontsize=15)
plt.ylabel('No of Movies', fontsize=15)
plt.xticks(index, label, fontsize=10, rotation=90)
plt.title('Market Share for Each Genre 1995-2017')
plt.show()




-----------------------------------------------------------------------

+++++++++++++++++++++++++++++++++++++++++
"""
Histogram - Bar Graph of Frequency 

"""
+++++++++++++++++++++++++++++++++++++++

"""






A histogram is used to summarize discrete or continuous data. 
In other words, it provides a visual interpretation of numerical data by 
showing the number of data points that fall within a specified range of 
values (called "bins").
However, a histogram, unlike a vertical bar graph, shows no gaps between the bars.


X Axis = width = Class Interval
Y Axis = height = Frequency

Creating a histogram provides a visual representation of data distribution.
The median and distribution of the data can be determined by a histogram.
It can show any outliers or gaps in the data.

Types of Distribution
1. Normal distribution
    Points on one side of the average are as likely to occur as on the other 
    side of the average

2. Bimodal distribution
    There are two peaks
    The data should be separated and analyzed as separate normal distributions
    
3. Right-skewed distribution
    A large number of the data values occur on the left side 

4. Left-skewed distribution
    A large number of the data values occur on the right side

5. Random distribution
    Has several peaks


https://corporatefinanceinstitute.com/resources/excel/study/histogram/
"""


"""
Ramesh is the branch manager at a local bank. 
Recently, Ramesh’s been receiving customer feedback saying that the wait times 
for a client to be served by a customer service representative are too long. 
Ramesh decides to observe and write down the time spent by each customer on waiting.

Write down the wait times spent by 20 customers
[43.1,35.6,37.6,45.3,43.5,40.3,50.2,47.3,31.2,42.2,45.5,30.3,31.4,35.6,45.2,
54.1,45.6,36.5,43.1]

"""




import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Customers wait times in seconds ( n = 20 customers )
customerWaitTime = [43.1,35.6,37.6,45.3,43.5,40.3,50.2,47.3,31.2,42.2,45.5,30.3,31.4,35.6,45.2,54.1,45.6,36.5,43.1]

customerWaitTime.sort()
# [1 to 35]      30.3, 31.2, 31.4                     [3]
# [35 to 40]     35.6, 35.6, 36.5, 37.6               [4]
# [40 to 45]     40.3, 42.2, 43.1, 43.1, 43.5         [5]
# [45 to 50]     45.2, 45.3, 45.5, 45.6, 47.3         [5]
# [50 to 55]     50.2, 54.1                           [2]

#Ramesh can conclude that the majority of customers wait between 35.1 and 50 seconds.

print(customerWaitTime)
 
plt.hist(customerWaitTime,bins=[25,30,35,40,45,50,55]) 

plt.axis([25, 60, 0, 6]) 
plt.xlabel('Seconds')
plt.ylabel('Customers')



'''Code Challenge 1'''
"""
For our first pie chart, 
the data we will plot describes the number of students
who choose different engineering majors at 
colleges in the US each year.

Discipline 	            Number of graduates
Civil Engineering 	    15,000 graduates
Electrical Engineering 	50,000 graduates
Mechanical Engineering 	45,000 graduates
Chemical Engineering 	10,000 graduates

"""

'''Code Challenge 2'''
"""
x = ['Nuclear', 'Hydro', 'Gas', 'Oil', 'Coal', 'Biofuel']
energy = [5, 6, 15, 22, 24, 8]

Plot a Bar Graph
"""    








=====================================================================================================

                                              NUMPY

================================================================================================





# Convert your list data to NumPy arrays
import numpy as np
print (np.__version__)


a = [0,1,2,3,4,5,6,7,8]
print (type(a))
print (a)  
# it always prints the values with comma seperated
# thats list


x = np.array( a ) 
print (type(x))

print (x)
# it always prints the values WITHOUT comma seperated , 
# thats ndarray


"""
Explain the ndarray data Structure Image
NumPy_NDArray_Data_Structure.png
"""

# to print the data type of the elements of array 
print (x.dtype)


# to print the dimension of the array 
print (x.ndim)

# to print the shape of the array 
# returns a tuple listing the length of the array along each dimension
# For a 1D array, the shape would be (n,) 
# where n is the number of elements in your array.
print (x.shape)


# Shows bytes per element 
print (x.itemsize)

# reports the entire number of elements in an array
print(x.size)

# returns the number of bytes used by the data portion of the array
print (x.nbytes)

print (x.strides)


# Array Indexing will always return the data type object 
print (x[0])

# Array Slicing
print (x[0:2])



"""
Reshaping is changing the arrangement of 
items so that shape of the array changes
Flattening, however, will convert a multi-dimensional 
array to a flat 1d array. And not any other shape.
"""

# Reshaping to 2 Dimensional Array - 3 Rows and 3 Columns
x = x.reshape(3,3)
print (x)


print (x.ndim)
print (x.shape)
print (x.strides)


# Due to reshaping .. none of the below has changed 
print (x.dtype)
print (x.itemsize)
print (x.size)
print (x.nbytes)


# Reshaping to 3 Dimensional Arry -  3 layers of 3 Rows and 3 Columns 
x = np.array( [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27] )
print(x)

print (x.ndim)
print (x.shape)
print (x.strides)

print (x.dtype)
print (x.itemsize)
print (x.size)
print (x.nbytes)


x = x.reshape(3,3,3)
 
print (x)
print (x.ndim)
print (x.shape)
print (x.strides)

print (x.dtype)
print (x.itemsize)
print (x.size)
print (x.nbytes)

       

"""
For 1D array, shape return a  tuple with only 1 component (i.e. (n,))
For 2D array, shape return a  tuple with only 2 components (i.e. (n,m))
For 3D array, shape return a  tuple with only 3 components (i.e. (n,m,k) )
"""


# Creating 2 Dimensional Array 
x = np.array( [ [1, 2, 3], [4, 5, 6] ] )
print (type(x))

print (x)
print (x.ndim)
print (x.shape)
print (x.dtype)
# For 2D array, return a shape tuple with only 2 elements (i.e. (n,m))



# Array Indexing 
print (x)

print (x[0])
print (type(x[0]))

print (type(x[0,0]))
print (x[0,0])
print (x[0,1])
print (x[0,2])


print (x[1])
print (type(x[1]))

print (type(x[1,0]))
print (x[1,0])
print (x[1,1])
print (x[1,2])



# Creating 3 Dimensional Array

#x = np.array([ [ [1, 2, 3], [4, 5, 6]], [ [11, 22, 33], [44, 55, 66] ], [ [111, 222, 333], [444, 555, 666] ]  ] )
x = np.array([ [ [1, 2, 3], [4, 5, 6], [7,8,9] ], [ [11, 22, 33], [44, 55, 66], [77,88,99] ], [ [111, 222, 333], [444, 555, 666], [777,888,999] ]  ] )
print (x)

print (x.ndim)
print (x.shape)
print (x.dtype)
# For 3D array, return a shape tuple with only 3 elements (i.e. (k,n,m) )
# We introduced the concept of another layer represented by n


# Array Indexing 
print (x)
print (x[0])      # 0th Layer
print (x[0,0])    # 0th Layer and 0th Row
print (x[0,0,0])  # 0th Layer and 0th Row and 0th Column



"""
Array         Dimen     Shape
[1 2 3]         1       (3,)

[[1 2 3]]       2       (1,3)

[[[1 2 3]]]     3       (1,1,3)

"""


"""
There are a couple of mechanisms for creating arrays in NumPy:
 a. Conversion from other Python structures (e.g., lists, tuples).
 b. Built-in NumPy array creation (e.g., arange, ones, zeros, etc.).
 c. Reading arrays from disk, either from standard or custom formats 
     (e.g. reading from a CSV file).
"""
# Using the built in function arange 

# Arange function will generate array from 0 to size-1 
# arange is similar to range function but generates an array , 
# where in range gives you a list of elements

import numpy as np 

x = np.arange(20, dtype=np.uint8)
print (x)

# zeros(shape) -- creates an array filled with 0 values with the specified shape.
# The default dtype is float64.
x = np.zeros((3, 3))
print (x)

# ones(shape) -- creates an array filled with 1 values. 

import numpy as np 

x = np.ones((3, 3), dtype=np.int8 )
print (x)

# linspace() -- creates arrays with a specified number of elements, 
# and spaced equally between the specified beginning and end values.

import numpy as np 
x = np.linspace(1, 4, 10, dtype = np.float) # try with float16,float32,float64
print (x)


import numpy as np 
#random.random(shape) – creates arrays with random floats over the interval [0,1].
x = np.random.random((2,3))*100
print (x)


"""
Arrays Operations - Basic operations apply element-wise. 
The result is a new array with the resultant elements.
Operations like *= and += will modify the existing array.
"""

import numpy as np
a = np.arange(5) 
print (a)

b = np.arange(5) 
print(b)


x= np.array(list(zip(a,b)))
print (x) 

x = a + b
print (x) 

x = a - b
print (x)

x = a**3
print (x)
 
x = a>3
print (x)
 
x= 10*np.sin(a)
print (x) 

x = a*b
print (x)


# Basic Functions         
import numpy as np 

x = np.array( [[1,2,3],[4,5,6]]) 

print("Sum is: ", x.sum())
print("Average/Mean value is: ", x.mean())
print("Max value is: ", x.max())
print("Min value is: ", x.min())

#print("Median value is: ", np.median(x))
#print("Correlation coefficient value is: ", np.corrcoef(x))
#print("Standard Deviation is: ", np.std(x))


# Row wise min
print("Row wise minimum: ", np.amin(x, axis=0))
print("Row wise maximum: ", np.amax(x, axis=0))


# Column wise min
print("Column wise minimum: ", np.amin(x, axis=1))
print("Column wise maximum: ", np.amax(x, axis=1))



"""
Mean, Median, Mode

Let's create some fake income data, centered around 27,000 
with a normal distribution and standard deviation of 15,000, with 10,000 data points. 
Then, compute the mean (average)

"""

import numpy as np
                          #mean, sd, total
incomes = np.random.normal(27000, 15000, 10000)
#loc=150, scale=20, size=1000

print (type(incomes))
print (incomes.size)
print (incomes)
print (len(incomes))
print (incomes.ndim)
print (incomes.shape)
print (incomes.dtype)

print("Mean value is: ", np.mean(incomes))
print("Median value is: ", np.median(incomes))


from scipy import stats
print("Mode value is: ", stats.mode(incomes)[0])
 

print("Minimum value is: ", np.min(incomes))
print("Maximum value is: ", np.max(incomes))
print("Standard Deviation is: ", np.std(incomes))
#print("Correlation coefficient value is: ", np.corrcoef(incomes))



#We can segment the income data into 50 buckets, and plot it as a histogram:
import matplotlib.pyplot as plt
plt.hist(incomes, 20)
plt.show()


#box and whisker plot to show distribution
#https://chartio.com/resources/tutorials/what-is-a-box-plot/
plt.boxplot(incomes)

# Explain NumPy_boxplot.png


print("Mean value is: ", np.mean(incomes))
print("Median value is: ", np.median(incomes))

#Adding Bill Gates into the mix. income inequality!(Outliers)
incomes = np.append(incomes, [10000000000])

#Median Remains Almost SAME
print("Median value is: ", np.median(incomes))

#Mean Changes distinctly
print("Mean value is: ", np.mean(incomes))

      

"""
Matplotlib with Numpy 
"""

# 2D Plot
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2*np.pi*t)
plt.plot(t, s)

plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('A nice sine wave')
plt.grid(True)
plt.savefig('sinewave.png')
plt.show()




# Normalised
import numpy as np
import matplotlib.pyplot as plt

data = np.random.randn(1000)

plt.hist(data, normed=True, bins=30)  

        



"""
Code Challenge
  Name: 
    Space Seperated data
  Filename: 
    space_numpy.py
  Problem Statement:
    You are given a 9 space separated numbers. 
    Write a python code to convert it into a 3x3 NumPy array of integers.
  Input:
    6 9 2 3 5 8 1 5 4
  Output:
      [[6 9 2]
      [3 5 8]
      [1 5 4]]
  
"""



"""
Code Challenge
  Name: 
    E-commerce Data Exploration
  Filename: 
    ecommerce.py
  Problem Statement:
      To create an array of random e-commerce data of total amount spent per transaction. 
      Segment this incomes data into 50 buckets (number of bars) and plot it as a histogram.
      Find the mean and median of this data using NumPy package.
      Add outliers 
          
  Hint:
      Execute the code snippet below.
      import numpy as np
      import matplotlib.pyplot as plt
      incomes = np.random.normal(100.0, 20.0, 10000)
      print (incomes)
 
    outlier is an observation that lies an abnormal distance from other values 
    
"""



"""
Code Challenge
  Name: 
    Normally Distributed Random Data
  Filename: 
    normal_dist.py
  Problem Statement:
    Create a normally distributed random data with parameters:
    Centered around 150.
    Standard Deviation of 20.
    Total 1000 data points.
    
    Plot the histogram using matplotlib (bucket size =100) and observe the shape.
    Calculate Standard Deviation and Variance. 
"""


"""
Code Challenge
  Name: 
    Random Data
  Filename: 
    random_data.py
  Problem Statement:
    Create a random array of 40 integers from 5 - 15 using NumPy. 
    Find the most frequent value with and without Numpy.
  Hint:
      Try to use the Counter class
      
"""






===================================================================================================

                                               PANDAS

================================================================================================




+++++++++++++++++++++++++++

Day1 - Data Exploring

+++++++++++++++++++++++++++



# Open salaries.csv and explain

"""
Analysis of Salaries Data

   How many are Male Staff and how many are Female Staff. 
   
   How many are Prof, AssocProf and AsstProf. 

   Who are the senior and junior most employees in the organization.

   Which Male Professor has the highest and the lowest salaries

   Which Female Professor has the highest and the lowest salaries
   
   Which Professor takes the highest and lowest salaries.

"""

# Solution Approach
"""
'''Loading your data '''
'''Exploring your data '''
'''How to select ROWS and COLUMNS from the DATA FRAME '''
'''Deep Exploration your data '''
'''Filtering - Using Conditionals to Filter Rows and Columns '''
'''Updating Columns Name '''
'''Modifying Row Data Within DataFrames '''
'''Add/Remove Columns From DataFrames '''
'''Add/Remove Rows From DataFrames '''
'''Sorting Data '''
'''Grouping and Aggregating '''
'''Casting Datatypes '''
'''Handling Missing Values '''
"""


# Open the image Pandas

'''Loading your data '''
import pandas as pd
df = pd.read_csv("Salaries.csv")


'''Exploring your data '''
# Not a good technique to print the Data Frame
# But this prints limited rows and columns
print (df)
pd.set_option("display.max_rows", None)
pd.set_option('display.max_columns', None)
print (df)

#number of dimensions
df.ndim   

#return a tuple representing the dimensionality
df.shape 

#number of elements
df.size 

#list the column names / column Indexes
df.columns 
df.columns.tolist()

# Gives the row Indexes
df.index
df.index.tolist()

#Check types for all the columns
df.dtypes


#list the row labels/index and column names
df.axes
df.axes[0]
df.axes[1]

#numpyrepresentation of the data
df.values 

df.info()

#List first 5 records
df.head()
df.head(10)

#Can you guess how to view the last few records;
df.tail()
df.tail(10)

#returns a random sample of the data frame
df.sample(5) 


# generate descriptive statistics (for numeric columns only)
# Standard Deviation is quite useful tool to figure out 
# how the data is spread above or below the mean.
# The higher the value, the less is reliable or vice versa. 
df.describe() # Numeric Columns
print(type(df.describe()))   
 

# In order to see statistics on non-numerical features,
# include = all
df.describe(include=['object', 'bool','float64','int64'])
df.describe(include=['object', 'bool'])
df.describe(include=['float64','int64'])
df.describe(include=['object','float64','int64'])


# This gives a missing values in salary and phd columns
# Finding missing data
df.isnull().any(axis=0)

# Finds rows which has missing values in any column
df.isnull().any(axis=1)

#return max/min values for all columns
df.max() 
df.min()

#return max/min values for all numeric columns
df.mean()
df.median()
df.std()

#What are the mean values of the first 50 records in the dataset?
df.head(50).mean()



'''How to select ROWS and COLUMNS from the DATA FRAME '''

# Selecting single column 
print(df['rank'])

print(df['discipline'])

# Its a series 
print(type(df['discipline']))

print(df.discipline)

print(df.rank)   # DO NOT USE THIS TECHNIQUE


# SELECT multiple columns
print(df[ ['rank', 'discipline'] ])

# How to get the Rows using the Integer Location / Position 
df.iloc[0]

df.iloc[ [0,1] ]

# How to get the specific Rows and specific columns
df.iloc[ [0,1], 0 ]

df.iloc[ [0,1], [0,1] ]

# We can use slicing for controlling the rows and columns
df.iloc[ 0:1, 0:1 ]   # u do not use [] when using slicing


# How to get the Rows using the Index 
df.loc[0]

df.loc[ [0,1] ]


# How to get the specific Rows and specific columns
df.loc[ [0,1] , 'rank']

df.loc[ [0,1] , ['rank', 'discipline']]

# We can use slicing for controlling the rows and columns
df.loc[:, 'rank' : 'discipline' ] # u do not use [] when using slicing


'''Deep Exploration your data '''

# Find unique values in a Series / Column
df['rank'].unique()
df['discipline'].unique()
df['sex'].unique()
list1 = df['sex'].unique().tolist()


# intuition about a column data
df['rank']
df['rank'].value_counts() # normalize = True


#calculate the basic statstics on the salary column
df['salary'].mean()
df['salary'].std()
df['salary'].describe()


#Find how many values in the salary column which are non NaN (use count method);
df['salary'].count()
df['phd'].count()


# Boolean Indexing
# Find those rows which has null values in salary/phd column
df['salary'].isnull()
df[df['salary'].isnull()]

df['phd'].isnull()
df[df['phd'].isnull()]
  

'''Filtering - Using Conditionals to Filter Rows and Columns'''

# select only those faculty who has salary more than 120000
filter_1 =  ( df['salary'] > 120000 )
print(filter_1)  # Return as Series of Boolean Values 

# Applying the filter to the data frame
# This concept is known as Boolean Indexing
print( df[filter_1] )
print( df.loc[filter_1] )
# to display only the selected column
print( df.loc[filter_1,'salary'] )


# Filtering those faculty who has more than 120000 and have more than 10 years 
# expereince and are female
# | for OR  ~ for NOT
 
filter_2 = (df['salary'] > 120000) & \
           (df['phd'] > 10) & \
           (df['sex'] == 'Female' )

df_sub= df[filter_2]
print(df_sub)
print(df.loc[filter_2])


#Select only those rows that contain female professors:
filter_3 = (df['sex'] == 'Female')  
df_sub= df[filter_3]
print(df_sub)
print(df.loc[filter_3])
print(df_sub[['salary','sex']])







---------------------------------------------------------------------------------------





+++++++++++++++++++++++++++++++++++++++++++

DAY 4 -    Data Updating

++++++++++++++++++++++++++++++++++++++++++




'''Loading your data '''
import pandas as pd
df = pd.read_csv("Salaries.csv")


'''Updating Columns Name '''
print(df.columns.tolist())

#old col name ['rank', 'discipline', 'phd', 'service', 'sex', 'salary']            
df.columns = ['post', 'department', 'phd', 'service', 'gender', 'salary']
print(df.columns.tolist())

# Renaming few of the colunm name 
df.rename(columns={'post':'rank', 'department':'discipline', 'sex':'gender'}, inplace = True )
print(df.columns.tolist())

# To update all the Column Name to UPPERCASE
df.columns = [x.upper() for x in df.columns ] 
print(df.columns.tolist())


# Getting back to what was old
df = pd.read_csv("Salaries.csv")
print(df.columns.tolist())


'''Modifying Row Data Within DataFrames '''
df.isnull().any(axis=0)  # Column Wise
df.isnull().any(axis=1)  # Row Wise
print(df[df.isnull().any(axis=1)])  # 7 13 28 34 

print(df.loc[7])
df.loc[7, ['salary']] = round(df['salary'].mean(),2)
print(df.loc[7])

# Another Advance Logic
df.loc[(df.service == 18)]
df.loc[(df.service == 18)]['salary']
df.loc[(df.service == 18)]['salary'].mean()
df.loc[7, ['salary']] = df.loc[(df.service == 18)]['salary'].mean()


print(df[df.isnull().any(axis=1)])  # 13 28 34 

print(df.loc[13])
df.loc[13, ['phd']] = round(df['phd'].mean(),0)
print(df.loc[13])

# Another Advance Logic
df.loc[(df.service == 33)]
df.loc[(df.service == 33)]['phd']
df.loc[(df.service == 33)]['phd'].mean()
df.loc[13, ['phd']] = df.loc[(df.service == 33)]['phd'].mean()


print(df[df.isnull().any(axis=1)])  # 28 34 


# Replace A with Computer Science and B with IT
# This keeps the non matching to old values only
df['discipline'] = df['discipline'].replace({'A':'Computer Science','B':'IT'})

# If there are any non matching it will convert that into NaN
df['discipline'] = df['discipline'].map({'A':'Computer Science','B':'IT'})


'''Add/Remove Columns From DataFrames '''
df = pd.read_csv("Salaries.csv")

df['full_name'] = df['rank'] + ' of department ' + df['discipline']

# Deleteing columns from the data frame
df.drop(columns=['full_name'], inplace=True)


# Creating a new Column based on some other columns values 
df = pd.read_csv("Salaries.csv")
# Male == 1 and Female == 0
df["bool_sex"] = df["sex"].map(lambda x: 1 if x == 'Male' else 0 )
print(df)

# Create a new column 
import numpy as np
df = pd.read_csv("Salaries.csv")
df['senior'] = np.where(df['service']>10, 'yes', 'no')
print(df)


'''Add/Remove Rows From DataFrames '''

# Addding a row in the data frame
df = df.append({'rank': 'Prof','discipline': 'A','phd': 25.0,
                'service': 15, 'sex': 'Male','salary': 109646.0 }, 
                ignore_index= True)

# Deleting a specific row from the Data Frame
df.drop(index=78, inplace = True)

# Deleting all Female AsstProf
df = df.drop(df[(df['sex']== 'Female') & (df['rank']== 'AsstProf')].index)


'''Sorting Data '''

# Create a new data frame from the original sorted by the column Salary
df_sorted= df.sort_values( by='salary')
df_sorted.head()

# To find the lowest salary of the employee
df_sorted= df.sort_values( by='salary', ascending = [True])
df_sorted.head(1)


#We can sort the data using 2 or more columns:
df_sorted= df.sort_values( by=['service','salary'], ascending = [True,True])
df_sorted.head(10)

df['salary'].nlargest(10)
df['salary'].nsmallest(10)


'''Grouping and Aggregating'''
#Split the data into groups based on some criteria

#Group data using rank
df = pd.read_csv("Salaries.csv")
df_rank= df.groupby(['rank'])

df_rank.size()
df_rank.get_group('AssocProf')
df_rank.groups


# Groups returns a dictionary object
df_rank.groups['AssocProf']
df_rank.groups['AssocProf'][0]
df.loc[6]
 
#group data using rank followed  by discipline and sex
df_rank=df.groupby(['rank', 'discipline','sex'])
df_rank.size()
df_rank.groups

 
#Calculate mean salary for each type of professor rank:
df.groupby(['rank'])[['salary','phd']].min()
df.groupby(['rank'])[['salary','phd']].max()
df.groupby(['rank'])[['salary','phd']].mean()
        

'''Casting Datatypes '''
# Casting into different data types
df.dtypes
df['service'] = df['service'].astype(float)
df.dtypes


'''Handling Missing Values '''
import numpy as np

# Finding missing data
df.isnull().any(axis=0)  # Column Wise
df.isnull().any(axis=1)  # Row Wise
df[df.isnull().any(axis=1)]


df.info()
df[df['phd'].isnull()]
df[df['salary'].isnull()]


df.dropna()
df.dropna(axis='index', how='any') 
df.dropna(axis='index', how='all') 
df.dropna(axis='column',how='all') 
df.dropna(axis='index', how='all', subset=['last', 'email'])

df.replace('NA', np.nan, inplace=True)
df.replace('Missing', np.nan, inplace=True)
# mark zero values as missing or NaN
df['salary'] = df['salary'].replace(0, np.NaN)

df.isna()

df.fillna('missing')
df.fillna(0)


# fill all the records with missing values, with mean of that column
df['phd'] = df['phd'].fillna(df['phd'].mean())

# fill all the records with missing values, with mean of that column
df['salary'] = df['salary'].fillna(df['salary'].median())


''' Iterating over rows '''
for i, row in df.iterrows():
    print("Index {}".format(i))
    print("Row \n{}".format(row))


''' Code Challenge '''

"""
Analysis of Salaries Data

1. Which Male Professor has the highest and the lowest salaries

1. Which Female Professor has the highest and the lowest salaries
   
2. Which Professor takes the highest and lowest salaries.

3. Missing Salaries - should be mean of the matching salaries of those 
   whose service is the same
   
4. Missing phd - should be mean of the matching service 

5. How many are Male Staff and how many are Female Staff. 
   Show both numbers and in percentage
   Show both in numbers and Graphically using Pie Chart.  
   
6. How many are Prof, AssocProf and AsstProf. 
   Show both in numbers and Graphically using a Pie Chart   

7. Who are the senior and junior most employees in the organization.

8. Draw a histogram of the salaries divided into bin starting 
   from 50K and increment of 15K
"""









