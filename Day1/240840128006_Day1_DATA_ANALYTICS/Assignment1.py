import numpy as np
from imageio.v2 import sizes

print("Ques_1")
print(np.__version__)

print(np.show_config())

print("Ques2")
help(np.add)



print("Ques3")

array = np.array([1, 2, 3, 4, 5])

none_zero = np.all(array != 0)

print("None of the elements is zero: ", none_zero)

print("Ques4")

arr = np.array([1, 2, 0, 4, 5])

any_Nonzero = np.any(array != 0)
print(any_Nonzero)




print("Ques5")
array = np.array([1, 2, np.inf, -np.inf, np.nan, 3])

finiteness = np.isfinite(array)

print("Array:", array)
print("Finiteness:", finiteness)

print("Ques6")
array = np.array([1, 2, np.inf, -np.inf, 3, 0])

positive_infinity = np.isposinf(array)

negative_infinity = np.isneginf(array)

print("Array:", array)
print("Positive Infinity:", positive_infinity)
print("Negative Infinity:", negative_infinity)


print("Ques7")

array = np.array([1,np.nan,2,np.nan,3,np.nan])
nan_check = np.isnan(array)
print(nan_check)

print("Ques8")

array = np.array([1,1j+2,3,6j-1,5])
complex_check = np.iscomplex(array)
real_check = np.isreal(array)
scalar_check = np.isscalar(array)
print("Complex:  ",complex_check)
print("Real: ",real_check)
print("Scalar: ",scalar_check)


print("Ques9")
array1 = np.array([1,2,8,4,5])
array2 = np.array([6,7,3,4,10])

greater = np.greater(array1,array2)
greater_equal = np.greater_equal(array1,array2)
less = np.less(array1,array2)
less_equal = np.less_equal(array1,array2)
print("Greater: ",greater)
print("Greater Equal: ",greater_equal)
print("Less: ",less)
print("Less Equal: ",less_equal)

print("Ques10")
array = np.array([1,7,13,105])
memorySize = array.nbytes
print("Array: ",array)
print("Memory Size: ",memorySize)


print("Ques11")

array_zero = np.zeros(10)
array_one = np.ones(10)
print("Array Zero: ",array_zero)
print("Array One: ",array_one)
array_five = np.full(10,5)
print("Array Five: ",array_five)

print("Ques12")
array = np.arange(30,70)
array1  =array.reshape(10,4)
print("Array: ",array)
print("Array: ",array1)

print("Ques13")
array = np.arange(30,70,2)
print("Array: ",array)

print("Ques14")
array_identity = np.eye(3)
print(array_identity)

print("Ques15")
array_random = np.random.randint(0,1)
#array_random1 = np.linspace(0,1,5)
print("Array random: ",array_random)
#print("Array random1: ",array_random1)

print("Ques16")
array_stdv = np.random.standard_normal(15)
print("Array Standard Deviation: ",array_stdv)

print("Ques17")
array = np.arange(15,55)


print("Array: ",array[1:])

print("Ques18")
array = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

print("Array 3*4: \n",array)

print("Ques19")

array_vector = np.linspace(5,50,10)
print("Array: ",array_vector)

print("Ques20")
array = np.arange(0,20)
array_sign_change = -array[9:15]
print("Array: ",array)
print("Array_Sign_change: ",array_sign_change)

print("Ques21")
array = np.random.randint(0,10,size = 5)
print("Array: ",array)

print("Ques22")
array1 = np.array([1,2,3,4,5])
array2 = np.array([6,7,8,9,10])
array_multi = np.multiply(array1,array2)
print("Array1 * Array2: ",array_multi)

print("Ques23")
array1 = np.array([10,11,14,17])
array2 = np.array([12,15,16,18])
array3 = np.array([13,19,21,20])
array_matrix = np.concatenate((array1,array2,array3))
print("Matrix: \n",array_matrix)
array_matrix = array_matrix.reshape(3,4)
print("Matrix 3*4: \n",array_matrix)

print("Ques24")
array = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print("Array_Shape: \n",array.shape)

print("Ques25")
array = np.array([[1,0,0],[0,1,0],[0,0,1]])

print("Array_identity: \n",array)

print("Ques26")
array = np.zeros((10,10))
print("Array:\n",array)
array[0,:] = 1
array[-1,:] = 1
array[:,0] = 1
array[:,-1] = 1
print("Array: \n",array)

print("Ques27")
array = np.zeros((5,5))
np.fill_diagonal(array,[1,2,3,4,5])
print("Array: \n",array)

print("Ques28")
array = np.random.rand(3,3,3)
print("Array: \n",array)

print("Ques29")
array1 = np.array([1,2,3,4,5])
array2 = np.array([6,7,8,9,10])
array_inner  = np.inner(array1,array2)
array_dot = np.dot(array1,array2)
print("Array_Inner: ",array_inner,"\nArray_Dot: ",array_dot)

print("Ques30")
array = np.array([[11,21,31],[4,51,6],[71,8,9],[10,11,12]])
array_sort_R = np.sort(array,axis = 0)
print("ArrayC: \n",array_sort_R)
array_sort_C = np.sort(array,axis = 1)
print("ArrayR: \n",array_sort_C)

print("Ques31")
array = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
array_greaterthan = array > 7
array_lessthan = array < 6
print("Array: \n",array)
print("Array_greaterthan 7: \n",array[array_greaterthan])
print("Array_lessthan 6: \n",array[array_lessthan])

print("Ques32")
array = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
arrayGreater = array > 7
arrayLess = array < 6
arrayEqual = array == 9
array[arrayGreater] = 99
array[arrayLess] = 88
array[arrayEqual] = 77
print("Array: \n",array)

print("Ques33")
array = np.array([[1,2,3],[4,5,6],[10,11,12]])
array1 = np.empty_like(array)
print("Array: \n",array1)

print("Ques34")
array_3d = np.full((3, 5, 4), fill_value="A", dtype=str)
print("Array: \n",array_3d)



print("Ques35")

arra1 = np.array([[1,2,3],[4,5,6]])
array2 = np.array([[7,8,9],[10,11,12]])
array = np.multiply(arra1,array2)
print("Array: \n",array)