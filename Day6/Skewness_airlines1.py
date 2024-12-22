import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

airline_df = pd.read_csv('airline_dec_2008_50k.csv')

distance = airline_df['Distance']

mean_distance = distance.mean()
median_distance = distance.median()
skewness_distance = distance.skew()

plt.figure(figsize = (8,6))

sns.histplot(distance, kde = True , color = 'skyblue', bins = 30)
plt.axvline(mean_distance,color = 'red', linestyle = '--',label =f'Mean: {mean_distance:.2f}')
plt.axvline(median_distance,color = 'black', linestyle = '--',label = f'Median: {median_distance:.2f}')
plt.title(f'Distribution of distance \n skewness: {skewness_distance:.2f}')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

airline_df1 = pd.read_csv('airline_passenger_satisfaction.csv')
print(airline_df1.columns)
satisfaction = airline_df1['On-board Service']


mean_satisfaction = satisfaction.mean()
median_satisfaction = satisfaction.median()
skewness_satisfaction = satisfaction.skew()

plt.figure(figsize = (8,6))

sns.histplot(satisfaction, kde = True , color = 'skyblue', bins = 30)
plt.axvline(mean_satisfaction,color = 'red', linestyle = '--',label =f'Mean: {mean_satisfaction:.2f}')
plt.axvline(median_satisfaction,color = 'black', linestyle = '--',label = f'Median: {median_satisfaction:.2f}')
plt.title(f'Distribution of rating \n skewness: {skewness_satisfaction:.2f}')
plt.xlabel('Online Boarding')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()