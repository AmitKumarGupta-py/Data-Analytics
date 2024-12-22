import os.path

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from openpyxl.styles.builtins import total
from sympy.printing.pretty.pretty_symbology import line_width

data = pd.read_csv("tips.csv")

#extract the total bill column into a new series

total_bill_series = data['total_bill']
print(total_bill_series.head())

#calculate mean and std
mu , std = total_bill_series.mean(), total_bill_series.std()

#calculate z-score for each value

z_scores = (total_bill_series - mu) / std
#plot normal distribution graph of total bill
plt.figure(figsize=(12,6))

#histogram

plt.subplot(1,2,1)

plt.hist(total_bill_series,bins = 20, density =True, alpha = 0.6, color = 'b', edgecolor = 'black')

#fit a normal distribution to the data

x = np.linspace(total_bill_series.min(),total_bill_series.max(),100)
p = stats.norm.pdf(x, mu, std)

plt.plot(x,p,'k',linewidth = 2)
title = "fit results : mu = %.2f, std = %.2f" % (mu,std)
plt.title(title)
plt.xlabel('Total Bill')
plt.ylabel('Density')
plt.grid(True)

#QQ plot
plt.subplot(1,2,2)
stats.probplot(total_bill_series,dist = "norm", plot = plt)
plt.title('QQ Plot')
plt.tight_layout()
plt.show()

#plot graph showing mean +- 3SD

plt.figure(figsize=(10,6))
plt.hist(total_bill_series,bins = 20, density = True, alpha = 0.6, color = 'b',edgecolor = 'black')

#Fit a normal distribution to the data

x =np.linspace(total_bill_series.min(),total_bill_series.max(),100)
p = stats.norm.pdf(x,mu,std)

plt.plot(x,p,'k', linewidth = 2)
title = "Fit results: mu = %.2f, std= %.2f" % (mu,std)

plt.title(title)
plt.xlabel('Toatal Bill')
plt.ylabel('Density')
plt.grid(True)

#plot mean +- 3SD

for i in range(1,4):
    plt.axvline(mu - i*std, color = 'r', linestyle = '--',linewidth = 2, label = f'Mean - {i} SD')
    plt.axvline(mu + i * std, color='r', linestyle='--', linewidth=2, label=f'Mean - {i} SD')

plt.legend()
plt.show()

#Check if mean +- 1Sd and mean +- 3SD satisfy the empirical rule

within_1sd = (z_scores <= 1) & (z_scores >= -1)
within_2sd = (z_scores <= 2) & (z_scores >= -2)
within_3sd = (z_scores <=3) & (z_scores >= -3)

percentage_within_1sd = np.sum(within_1sd) / len(total_bill_series) * 100
percentage_within_2sd = np.sum(within_2sd) / len(total_bill_series) * 100
percentage_within_3sd = np.sum(within_3sd) / len(total_bill_series) * 100

print(f"Percentage of the data within mean +- 1 SD: {percentage_within_1sd:.2f}%")
print(f"Percentage of the data within mean +- 2 SD: {percentage_within_2sd:.2f}%")
print(f"Percentage of the data within mean +- 3 SD: {percentage_within_3sd:.2f}%")


#create DataFrame to store total_bill, z_score  , and within_1sd within 2sd within_3_sd flags

result_df = pd.DataFrame({
    'total_bill' : total_bill_series,
    'z_score' : z_scores,
    'within_1sd' : within_1sd,
    'within_2sd' : within_2sd,
    'within_3sd' : within_3sd

})

print(result_df.head())

if os.path.exists("deletethis-2.csv"):
    os.remove("deletethis-2.csv") #remove the file if it exists

result_df.to_csv("deletethis-2.csv", index= False)

#filter outliers
#outliesrs = result_df[(result_df['z_score']<-3) | (result_df['z_score'] > 3)]

#outliers z- score box plot

import seaborn as sns

plt.figure(figsize=(8,6))
#vert = False indicate that the boxplot will be drawn horizontally

#markers = '*' : sets the marker style for the outlier to a star('*')

plt.boxplot(result_df['z_score'],vert = False, flierprops= dict(marker = '*',markerfacecolor = 'red',markersize = 10))
plt.xlabel('Z-Score')
plt.title('Boxplot of Z-scores')
plt.grid(True)
plt.show()