import numpy as np
from sympy.matrices.expressions.slice import slice_of_slice

#create sample array
arr = np.arange(0,11)
print(arr)

#Bracket indexing and selection
#get a value at an index
print(arr[8])
#get values in a range
print(arr[1:5])

#get values in a range
print(arr[0:5])


#reset array,why? will be clean soon
arr = np.arange(0,11)
print(arr)

#important  notes on slice

sliceof_arr = arr[0:6]

print(sliceof_arr)

#Change slice
sliceof_arr[:] = 99

print(sliceof_arr)
print(arr)

#change made are also there in our original array
#data is not copied ,it is a view of the original array
#this avoids memory problems

#to get a copy , we need to be explicit
arr_copy = arr.copy()
print(arr_copy)


#conditional selection
arr = np.arange(1,11)
print(arr)

#check each elements of the array against the condition
#which returns a boolean array where
#each elements is true if the corresponding elemnts in arr>4
#and false otherwise

print( arr > 4)

#store boolean results in another array
bool_arr = arr > 4
print(bool_arr)

#select only those elements from the arr array where
#the corresponding elemnts in bool_arr is True
#if effectively filters out the elements of arr where the condition arr > 4 is true

print( arr[bool_arr])