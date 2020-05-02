# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:34:00 2020

@author: user
"""

"""

Code Challenge:
Dataset: Market_Basket_Optimization.csv
Q2. In today's demo sesssion, we did not handle the null values before 
fitting the data to model, 
remove the null values from each row and perform the associations once again.
Also draw the bar chart of top 10 edibles.


"""


import pandas as pd
from apyori import apriori

# Data Preprocessing
# Column names of the first row is missing, header - None
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

dataset.isnull().any(axis=0)

dataset = dataset.drop(dataset.iloc[:,7:],axis=1)

dataset.dropna(inplace=True)

transactions = []
for i in range(0, 1369):
    #transactions.append(str(dataset.iloc[i,:].values)) #need to check this one
    transactions.append([str(dataset.values[i,j]) for j in range(0, 7)])
    
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.25, min_lift = 4)

print(type(rules))

# Visualising the results
results = list(rules)
print(len(results))
results[0]
results[0].items
results[0][0]


for item in results:  #32 results
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")



------------------------------------------------------------------




"""
#solution


+++++++++++++++++++++++++++++++++++++++




Rule: brownies -> yogurt cake
Support: 0.005113221329437546
Confidence: 0.28
Lift: 6.084444444444445
=====================================
Rule: cookies -> shallot
Support: 0.004382761139517896
Confidence: 0.6666666666666666
Lift: 15.735632183908045
=====================================
Rule: escalope -> mushroom cream sauce
Support: 0.004382761139517896
Confidence: 0.3
Lift: 5.55
=====================================
Rule: mint green tea -> french fries
Support: 0.005843681519357195
Confidence: 0.7999999999999999
Lift: 4.071375464684015
=====================================
Rule: shallot -> green tea
Support: 0.003652300949598247
Confidence: 0.5555555555555556
Lift: 5.511272141706924
=====================================
Rule: protein bar -> hand protein bar
Support: 0.003652300949598247
Confidence: 0.38461538461538464
Lift: 22.892976588628763
=====================================
Rule: pasta -> shrimp
Support: 0.015339663988312637
Confidence: 0.7777777777777778
Lift: 4.192038495188101
=====================================
Rule: pet food -> red wine
Support: 0.004382761139517896
Confidence: 0.4615384615384615
Lift: 6.1945701357466065
=====================================
Rule: cookies -> burgers
Support: 0.005113221329437546
Confidence: 0.46666666666666673
Lift: 4.629468599033817
=====================================
Rule: cookies -> ham
Support: 0.004382761139517896
Confidence: 0.4
Lift: 5.587755102040816
=====================================
Rule: pasta -> burgers
Support: 0.007304601899196494
Confidence: 0.9090909090909093
Lift: 4.899785254115963
=====================================
Rule: escalope -> chocolate
Support: 0.003652300949598247
Confidence: 0.3125
Lift: 5.78125
=====================================
Rule: tomato sauce -> chocolate
Support: 0.005843681519357195
Confidence: 0.29629629629629634
Lift: 6.337962962962964
=====================================
Rule: cookies -> french fries
Support: 0.003652300949598247
Confidence: 0.8333333333333334
Lift: 4.241016109045849
=====================================
Rule: eggs -> grated cheese
Support: 0.003652300949598247
Confidence: 0.625
Lift: 4.945809248554913
=====================================
Rule: eggs -> pasta
Support: 0.003652300949598247
Confidence: 1.0
Lift: 5.389763779527559
=====================================
Rule: honey -> fresh tuna
Support: 0.005113221329437546
Confidence: 0.5833333333333334
Lift: 7.605555555555555
=====================================
Rule: turkey -> fromage blanc
Support: 0.003652300949598247
Confidence: 0.7142857142857143
Lift: 9.312925170068027
=====================================
Rule: low fat yogurt -> red wine
Support: 0.003652300949598247
Confidence: 0.33333333333333337
Lift: 4.473856209150328
=====================================
Rule: pancakes -> honey
Support: 0.003652300949598247
Confidence: 0.7142857142857143
Lift: 4.233147804576376
=====================================
Rule: milk -> pasta
Support: 0.005843681519357195
Confidence: 0.888888888888889
Lift: 4.790901137357831
=====================================
Rule: mineral water -> pasta
Support: 0.008035062089116142
Confidence: 1.0
Lift: 5.389763779527559
=====================================
Rule: ham -> eggs
Support: 0.003652300949598247
Confidence: 0.29411764705882354
Lift: 4.108643457382954
=====================================
Rule: french wine -> spaghetti
Support: 0.003652300949598247
Confidence: 0.25
Lift: 4.2253086419753085
=====================================
Rule: herb & pepper -> grated cheese
Support: 0.004382761139517896
Confidence: 0.5
Lift: 4.074404761904762
=====================================
Rule: light cream -> mineral water
Support: 0.004382761139517896
Confidence: 0.6666666666666666
Lift: 4.906810035842294
=====================================
Rule: spaghetti -> tomato sauce
Support: 0.003652300949598247
Confidence: 0.35714285714285715
Lift: 7.639508928571429
=====================================
Rule: olive oil -> tomatoes
Support: 0.003652300949598247
Confidence: 0.7142857142857143
Lift: 4.178876678876679
=====================================
Rule: parmesan cheese -> frozen vegetables
Support: 0.003652300949598247
Confidence: 0.2631578947368421
Lift: 4.803508771929825
=====================================
Rule: parmesan cheese -> tomatoes
Support: 0.003652300949598247
Confidence: 1.0
Lift: 4.038348082595871
=====================================
Rule: tomato sauce -> mineral water
Support: 0.003652300949598247
Confidence: 1.0
Lift: 4.038348082595871
=====================================
Rule: parmesan cheese -> tomatoes
Support: 0.003652300949598247
Confidence: 0.29411764705882354
Lift: 5.368627450980393
=====================================


++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


