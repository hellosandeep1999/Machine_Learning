# -*- coding: utf-8 -*-
"""
Created on Fri May  1 23:58:29 2020

@author: user
"""

"""
Q3 Data: "data.csv"

This data is provided by The Metropolitan Museum of Art Open Access
1. Visualize the various countries from where the artworks are coming.
2. Visualize the top 2 classification for the artworks
3. Visualize the artist interested in the artworks
4. Visualize the top 2 culture for the artworks
"""


-------------------------------------------------------------------------------------------


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('data.csv')

dataset = dataset.dropna(axis=1,how='all')

dataset.isnull().any(axis=0)

---------------------------------------------------------------------

"""

1. Visualize the various countries from where the artworks are coming.

"""

a = list(dataset["Country"].unique())
print("Various countries List")
print("============================")

for i,j in enumerate(a):
    print(i,"  ->  ",j)
    print()
    
print("============================")


--------------------------------------------------------------------




"""

2. Visualize the top 2 classification for the artworks


"""

list_classification = dict(dataset["Classification"].value_counts(dropna=False))

i = 0
for item,value in list_classification.items():
    if i == 2:
        break
    print(i+1," ",item," -> ",value)
    i += 1
    
    
-----------------------------------------------------------------------
   





 
"""
   3. Visualize the artist interested in the artworks
   
"""

print(len(dataset[dataset["Artist Role"] == "Artist"]))



-------------------------------------------------------------------------


"""

4. Visualize the top 2 culture for the artworks

"""

dataset = dataset.dropna(how='any', subset=["Culture"])
list_Culture = dict(dataset["Culture"].value_counts(dropna=False))

i = 0
for item,value in list_Culture.items():
    if i == 2:
        break
    print(i+1," ",item," -> ",value)
    i += 1
    
 

    
    















