from mpmath.calculus.extrapolation import nprod

import numpy as np
import pandas as pd

df_airbnb = pd.read_csv("AB_NYC_2019.csv")
print(df_airbnb.head(3))

#make id column the index
df2 = df_airbnb.set_index('id')

print(df2.head(3))

print(df2.name[3647])

#set pandas dipslay options show all columns
numeric_columns = df2.select_dtypes(include = ['float64','int64'])

numeric_columns['room_type'] = df2['room_type']

df3 = numeric_columns.groupby("room_type").mean()

pd.set_option('display.max_columns', None)
print(df3.head(3))
print(df3.index)

#note that room _type has become an index- we again convert it into a normal column

df3 = df3.reset_index()

pd.reset_option('display.max_columns') #also reset the column display to default
print(df3.head(3))


#row
#get a single row
#integer based
print(df_airbnb.iloc[0])

#name based
#First set the index to the column which will be used in locating the row

df = df_airbnb.set_index('host_name')
print(df.loc['John'])

#Rove a row

print(df.head())

df = df.drop('John',axis = 0)

print(df.head())

