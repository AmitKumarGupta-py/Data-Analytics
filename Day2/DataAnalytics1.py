import numpy as np

my_list = [1, 2, 3]
my_array = np.array([1,2,3])

print(type(my_list))

#Create a numpy array from a list

print(np.array(my_list))

#or From a list of list

my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
print(np.array(my_matrix))

#Create arrays using built in fuctions
#returns evenly Spaced values within a given interval
#start,stop,step
print(np.arange(0,10))
print(np.arange(0,11,2))

#Generate arrays of zeros or ones
print(np.zeros(3))
#print(np.zeros(5,5))
print(np.ones(3))
#print(np.ones(3,3))


#return evenly spaced numbers over a specified interval
#start , stop, number of elements (and not step)

#unlike numpy.arange(),the stop value is inluded in the result
#the Spacing b/w values is automatically determined
#specified number of values (num)

print(np.linspace(0,10,3))
print(np.linspace(0,5,20))


#random number arrays

print(np.random.rand(3))
print(np.random.rand(3,3))

#from normal distribution
print(np.random.randn(3))
print(np.random.randn(3,3))

#random integers from low (inclusive) to high(excluded)

print(np.random.randint(0,100))#single random integers b/w 1 and 100
print(np.random.randint(1,100,10))#10 random integer b/w 1 and 100

#seeding for reproducable results
np.random.seed(77)
print(np.random.rand(4))#4 random numbers

#Storing
arr = np.arange(25)#numbers from 0 to 24

ranarr = np.random.randint(0,50,10)#10 random numbers b/w 0 ND 50

print(arr)
print(ranarr)

#reshape: Returns an array containing the same data with a new shape
print(arr.reshape(5,5))

#MAx min and their index positions

print(ranarr.max())
print(ranarr.argmax())#position of max elements
print(ranarr.min())
print(ranarr.argmin())

