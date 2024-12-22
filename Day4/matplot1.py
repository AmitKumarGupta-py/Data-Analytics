import matplotlib.pyplot as plt
import numpy as np
a = np.linspace(0,10,11)
b = a**4
x = np.arange(0,10)
y = 2*x
print(x)
print(y)

#create an empty canvas
fig = plt.figure()
axes = fig.add_axes([0,0,1,1])#LEFT , BOTTOM WIDTH HEIGHT(range 0 to 1)

axes.plot(x,y)

plt.show()

fig = plt.figure()
axes = fig.add_axes([0,0,1,1])#left bottom width height(range 0 to 1)
axes.plot(a,b)
plt.show()

# ############ Sub plots ########################

''''fig, axes = plt.subplots()
axes.plot(x,y,'r')#r = red color
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('y=2x')
plt.show()

axes.plot(a,b,'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('y=2x')
plt.show()'''

fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(12,8))

#PARAMETERS AT THE AXES LEVEL
axes[0,0].plot(a,b)
axes[1,1].plot(x,y)
axes[0,1].plot(y,x)
axes[1,0].plot(b,a)

#use left right top bottom to stretch subplots
#use wspace,hspace to add spacing btween subplots

fig.subplots_adjust(left = None,right = None, top = None, bottom = None,wspace=0.5,hspace=0.5,)


plt.show()