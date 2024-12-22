import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas.core.config_init import styler_escape

#Generate random data

np.random.seed(42)

x = np.random.rand(100)
y = np.random.rand(100)

#Create scatter plot with Matplotlib]

plt.figure(figsize=(8,6))
plt.scatter(x,y,color = 'blue', alpha = 0.6)
plt.title('Scatter plot using Matplotlib',fontsize = 15)
plt.xlabel('X-axis',fontsize = 12)
plt.ylabel('Y-axis',fontsize = 12)
plt.grid(True)
plt.show()

#Create scatter plot with seaborn

sns.set(style = "whitegrid") # Set Style with white grid background
plt.figure(figsize=(8,6))
sns.scatterplot(x=x,y=y)
plt.title('Scatter plot using Seaborn',fontsize = 15)
plt.show()

#Line plot
