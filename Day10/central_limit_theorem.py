import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro

df = pd.read_csv('diabetes.csv')

print(df.head())

#shapiro test om Glucose column
shapiro_test = stats.shapiro(df['Glucose'])
print("Shapiro-Wilk p-value:",shapiro_test.pvalue)

if shapiro_test.pvalue < 0.05:
    print("The data likely does not follow a normal distribution.")
else:
    print("The data may be normally distributed, but the q-q plot can provide further insights.")


#generate a q-q plot for the sample means
stats.probplot(df['Glucose'],dist = "norm", plot = plt)
plt.title('Q-Q plot of the sample means')
plt.show()
#central limit theorem

sample_means = []
n_samples = 100
sample_size = 30

for _ in range(n_samples):
    sample = df['Glucose'].sample(n=sample_size,replace = True)
    sample_means.append(sample.mean())

#Plot the distribution of sample means
plt.hist(sample_means,bins = 30,edgecolor = 'k', alpha = 0.7)
plt.title('Distribution of sample means (n=30)')
plt.xlabel('Sample mean')
plt.ylabel('Frequency')
plt.show()

#generate a Q-Q plot for the sample means
stats.probplot(sample_means,dist="norm",plot=plt)
plt.title('Q-Q plot  of Sample Means')
plt.show()

#perform the Shapiro -Wilk test on the sample means

shapiro_test = stats.shapiro(sample_means)
print("Shapiro-Wilk p-value:",shapiro_test.pvalue)

#Interpret the result

if shapiro_test.pvalue < 0.05:
    print("The sample data likely does not follow a normal distribution.")
else:
    print("The sample data may be normally distributed, but the q-q plot can provide further insights.")

