#line plot

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("heart.csv")
ages = df.groupby("age").median().reset_index()#group by ages then respective to the grouped ages find median for all other
                                            #columns and then reset index to show properly
                                            #1: Group on the age column
                                            #2: Calculate median for each numeric column on the age group
                                            #3: Reset index to reset to numeric index

print(df.head())
print(df.info())

#Simple pandas

ages.plot.line("age","chol")
plt.show()

#matplotlib
#hrrps://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
fig, ax = plt.subplots()
ax.plot(ages["age"],ages["chol"], ls = ":", lw = 1.7)
ax.set_xlabel("Age")
ax.set_ylabel("Cholestrol")
ax.set_title("Cholestrol vs Age")
plt.show()

#seaborn

sns.lineplot(x= "age", y = "chol", data = ages,linestyle = ":",linewidth = 1.7)
plt.xlabel("Age")
plt.ylabel("Cholestrol")
plt.title("Cholestrol vs Age")
plt.show()
