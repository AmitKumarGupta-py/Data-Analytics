import numpy as np
import pandas as pd

#Creating a series from a python list

myindex =['USA','Canada','England']
mydata = [1776,1867,1821]

#Just the numeric index

myser = pd.Series(data = mydata)

print(myser)

#Now the named index

myser = pd.Series(data = mydata,index = myindex)
print("\n",myser)


#creating a series from Numpy Array
#First create a Numpy array using the earlier list

ran_data = np.random.randint(0,100,4)
print("\n",ran_data,"\n")
names = ['Alice','Bob','Charlie','Dave']
ages = pd.Series(ran_data, names)
print(ages)

#Creating  A series from a dictionary
ages = {'Sammy':5,'Frank':10,'Spike':7}
print(ages)
print(pd.Series(ages))

#using named index
q1 = {'Japan':80,'China':450,'India':200,'USA':250}
q2 = {'Brazil':100,'China':500,'India':210,'USA':260}

#Convert into Panda Series
sales_Q1 = pd.Series(q1)
sales_Q2 = pd.Series(q1)
print(sales_Q1)

#call values based on named index

print(sales_Q1['Japan'])
#integer based location info also retained
print(sales_Q1[0])

#be careful with potetial errors
#wrong name
#print(sales_Q1['France']

#accidental Extra space
#print(sales_Q1['USA ']


#grab just the index values
print(sales_Q1.keys())

#can perform operations broadcasted across entire Series
print(sales_Q1 * 2)
print(sales_Q2/100)

#Notice how pandas informs you of mismatch with NAn

print(sales_Q1 + sales_Q2)

#you can fill NAN with aany matching data type value you want

print(sales_Q1.add(sales_Q2,fill_value=0))