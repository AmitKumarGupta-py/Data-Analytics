import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dm_office_sales.csv")

##Countplot: A simple plot, it merely shows the total count of rows per category
#A histogram across a categorical , instead of quantitive , variable

plt.figure(figsize=(10,4),dpi = 200) #dpi = dots per inch of space
sns.countplot(x = 'division',data = df)
plt.show()

plt.figure(figsize=(10,4),dpi = 200)
sns.countplot(x = 'level of education',data = df)
plt.show()

#Breakdown within another category with 'hue

plt.figure(figsize=(10,4),dpi = 200)
sns.countplot(x = 'level of education',hue = 'training level',data = df)
plt.show()

#Using palette
plt.figure(figsize=(10,4),dpi = 200)
sns.countplot(x = 'level of education',data = df,hue = 'training level',palette = 'Set1')
plt.show()

plt.figure(figsize=(10,4),dpi = 200)

#paired would be a good choice  if there was a distinct jump from 0 and 1 to 2 and 3

sns.countplot(x = 'level of education',data = df,hue = 'training level',palette = 'Paired')
plt.show()

