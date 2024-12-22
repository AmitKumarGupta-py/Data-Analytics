import pandas as pd
import numpy as np

# creating a series from a python list

myindex = ['USA', 'INDIA', 'CHINA']
mydata = [1776, 1867, 1000]

# Just the numeric index

myser = pd.Series(data=mydata, index=myindex)
print(myser)

# series from numpy arr
names = ["Alice", "Blud", "Chad"]
ages = pd.Series(np.random.randint(0, 100, 3), names)
print(ages)

# series from dictionary
ages = {'Sammy': 5, 'Frank': 10, 'Spike': 7}
print(ages)
print(pd.Series(ages))

# using named index
q1 = {'Japan': 80, 'China': 450, 'India': 200, 'USA': 230}
q2 = {'Brazil': 180, 'China': 550, 'India': 210, 'USA': 270}

# convert to pandas series
sales_Q1 = pd.Series(q1)
sales_Q2 = pd.Series(q2)
print(sales_Q1)

# call values based on named index
print(sales_Q1['Japan'])
# integer based location information also retained!
print(sales_Q1.iloc[0]) # using iloc is better

# Accidental extra space
# print(sales_Q1['USA'])
# text case mistake
# print(sales_Q1['usa'])

# Series ops
# Grab the index keys
print(sales_Q1.keys())

# Can perform operations broadcasted across the entire Series
print(sales_Q1*2)
print(sales_Q2/100)

# Notice how pandas inform you of mismatch with NaN
print(sales_Q1 + sales_Q2)

# You can fill NaN with any matching data type value you want
print(sales_Q1.add(sales_Q2, fill_value=0))
