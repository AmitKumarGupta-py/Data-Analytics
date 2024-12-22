import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

#Generate x axis values (1to9)

x = np.arange(1,10)

# generate y axis values(1 to 18)
y = np.arange(1,19)

#select the first 9 elements from y to match x axis length

y = y[:9] # Slicing to get eleemnts from index 0(included) to 9(excluded)

#create  the line plot

plt.plot(x,y)
plt.xlabel('X Axis Title here')
plt.ylabel('Y Axis Title here')
plt.title('Plot Title here')
plt.scatter(x,y,s=50) #increase maker size to 50

plt.show()


#exporting a plot
plt.plot(x,y)

plt.savefig('example.png')



