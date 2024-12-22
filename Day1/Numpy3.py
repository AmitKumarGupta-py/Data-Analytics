import numpy as np
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

arr = np.arange(0,10)
print(arr)

#basic arithmetic

print(arr + arr)
print(arr * arr)
print(arr - arr)

#this will raise a warning on division by zero , but not an error
#it just fills the spot with nan
print(arr/arr)

#also a warning (but not an error ) relating to infinity
print(1/arr)
print(arr**3)

#Universal functions

print(np.sqrt(arr))
print(np.exp(arr))
print(np.sin(arr))
print(np.log(arr))


#summary statistics

print(arr.sum())
print(arr.mean())
print(arr.max())
print(arr.min())
print(arr.std())
print(arr.var())

#2D arrays
#This is a 2D array with 3 rows and 4 columns
arr_2d = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(arr_2d)

#Row and column count
print(arr_2d.shape)# shape is a property not a function


#sum all the columns for each row
print(arr_2d.sum(axis = 0))

#summ all the rows for each column
print(arr_2d.sum(axis = 1))
