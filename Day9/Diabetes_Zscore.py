import pandas as pd
import scipy.stats as stats
from scipy.stats import zscore

df = pd.read_csv('diabetes.csv')

#calculate mean and standard deviation of the glucose column

glucose_mean = df['Glucose'].mean()
glucose_std = df['Glucose'].std()

print(f"Mean of glucose:{glucose_mean}")
print(f"Standard Deviation of glucose:{glucose_std}")

glucose_value = 168

z_score_168_manual = (glucose_value - glucose_mean) / glucose_std
print(f"Z-score for glucose level of 168 :{z_score_168_manual}")


#zcore function
#add column
df['Glucose_zscore'] = zscore(df['Glucose'])
print(df)

#find the row index where glucose = 168
#df['Glucose'] == glucose_value = Returns "True" where Glucose = 168
#df[now gets the actual records for 168

#index returns the indices of rows matching  this condition

row_index = df[df['Glucose'] == glucose_value].index

#now we extract the first row from glucose _zscore df where Glucose = 168
#hence row_index[0] ... remember , we may have multiple row matching glucose = 168

z_score_168_using_function = df['Glucose_zscore'].iloc[row_index[0]]

print(f"Z-score for glucose level 168 (using zscore function): {z_score_168_using_function}")

#calculAte the percentile for this z_score
percentile_168 = stats.norm.cdf(z_score_168_manual) * 100  # cumulative probability(using stats.norm.cdf()) *  # multiply 100 to get percentage

print(f"Percentile for Glucose level of 168: {percentile_168}%")

#find the z_score at the 30th percentile

z_score_30 = stats.norm.ppf(0.30) #population probability function (stats.norm.ppf())
print(f"Z_score at the 30th percentile : {z_score_30}")

#calculate  the corresponding Glucose value for the z_score of the 30th percentile

glucose_value_30 = glucose_mean + z_score_30 * glucose_std

print(f"Gucose value corresponding to 30th percentile :{glucose_value_30}")

#given z_score of 2.8, calculate the corresponding Glucose level
z_score_given = 2.8
glucose_value_given = glucose_mean + z_score_given * glucose_std

print(f"Glucose value for z-score of 2.8:{glucose_value_given}")

#calculate the percentile for the z-score of 2.8

percentile_given = stats.norm.cdf(z_score_given) * 100 #to get percentage

print(f"percentage for z_score of 2.8: {percentile_given}%")

