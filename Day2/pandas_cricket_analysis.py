import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dask.array.numpy_compat import divide

df = pd.read_csv('MS_Dhoni_ODI_record.csv')
print(df.describe())
print(df.info())

# data cleaning: remove 'v ' from opposition column and get just the country name
df["opposition"] = df["opposition"].str.replace('v ', '', regex=False)
# df['opposition'] = df['opposition'].apply(lambda x: x[2:])  # same function as the above operation

# add a feature - 'year' column using the match 'date' column
df["date"] = pd.to_datetime(df['date'], dayfirst=True)
df['year'] = df['date'].dt.year.astype(int)
print(df['year'])

# batting average of ms dhoni in that match = total runs/(innings count-not out count)
# you can take the count of not outs using where and sum or just take count of a new column of 'not_out'
df["not_out"] = np.where(df['score'].str.endswith('*'), 1, 0)

# dropping columns that won't help with data analysis(here it is odi_number)
df.drop(columns='odi_number', inplace=True)

# creating a new df by dropping DNB and TDNB values in score and start with runs_scored
df_new = df.loc[((df['score'] != 'DNB') & (df['score'] != 'TDNB')), 'runs_scored':]  # second argument takes all columns after 'runs_scored'

# fixing data types of numerical columns
df_new['runs_scored'] = df_new['runs_scored'].astype(int)
df_new['balls_faced'] = df_new['balls_faced'].astype(int)
df_new['strike_rate'] = df_new['strike_rate'].astype(float)
df_new['fours'] = df_new['fours'].astype(int)
df_new['sixes'] = df_new['sixes'].astype(int)
# this is the end of the data cleaning phase
# print(df_new.info())
# print(df_new.head())
# print(df_new.tail())

# career stats of dhoniblud
first_match_date = df['date'].dt.date.min().strftime('%B %d, %Y')  # first match
print(f'First match date: {first_match_date}')
last_match_date = df['date'].dt.date.max().strftime('%B %d, %Y')  # last match
print(f'Last match date: {last_match_date}')

number_of_matches = df.shape[0]  # number of rows = matches played in his career
print(f'Number of total matches: {number_of_matches}')

number_of_innings = df_new.shape[0]  # taking new df where dnb and tdnb were removed
print(f'Number of innings: {number_of_innings}')

not_outs = df['not_out'].sum()  # number of not outs in career
print(f'Number of not outs in career: {not_outs}')

runs_scored = df_new['runs_scored'].sum()  # total career runs use df_new since dnb and tdnb have been removed from it
print(f'Total runs scored in career: {runs_scored}')

balls_faced = df_new['balls_faced'].sum()  # total balls faced

career_sr = (runs_scored / balls_faced)  # career strike rate
print(f'Total career strike rate: {career_sr:.3f}')

career_avg = (runs_scored / (number_of_innings - not_outs))  # career avg runs
print(f'Average runs scored per match: {career_avg:.3f}')

# for centuries scored
print(
    f"Centuries scored: {(df_new['runs_scored'] > 100).sum()}")  # (df_new['runs_scored']>100) is a boolean expression returning boolean vals in Series which satisfy the condition

# number of fours scored
print(f"Career fours scored: {df_new['fours'].sum()}")

# number of sixes scored
print(f"Career sixes scored: {df_new['sixes'].sum()}")

# highest score of career
print(f"Highest score of career: {df_new['runs_scored'].max()} runs against {df_new.loc[df_new['runs_scored'].idxmax(), 'opposition']}")

# visualization time
# number of matches played against different oppositions
# count occurrences of unique value in 'opposition' column
opposition_counts = df['opposition'].value_counts()
# plotting a bar graph
opposition_counts.plot(kind='bar', title='Number of matches against different oppositions', figsize=(8, 5))
plt.show()

# runs scored against each team by grouping dataframe by 'opposition' column
grouped_by_opposition = df_new.groupby('opposition')
# sum the 'runs_scored' column for each group
sum_of_runs_scored = grouped_by_opposition['runs_scored'].sum()
# sum_of_runs_scored is a series with opposition as index and runs as values. Convert it into a df and reset_index
runs_scored_by_opps = pd.DataFrame(sum_of_runs_scored).reset_index()
print(runs_scored_by_opps)
# plotting the graph
runs_scored_by_opps.plot(x='opposition', kind='bar', title='Runs scored against every team', figsize=(8, 5))
plt.show()

#Box plot
sns.boxplot(x='opposition', y='runs_scored', data=df_new,palette= 'Paired') #box plot highlights Outliers: Extreme data point
plt.show()

#looks crowded - Let us retain only major countries

#List of oppositions to filter

opposition_list =['England','Australia','West Indies','South Africa','New Zealand','Pakistan','Sri lanka','Bangladesh']

#Filter rows where 'opposition' is in the list

df_filtered = df_new[df_new['opposition'].isin(opposition_list)]

#Sort the filtered DataFrame in descending order of 'runs_scored'

df_filtered_sorted = df_filtered.sort_values(by='runs_scored',ascending=False)

#Display the filtered DataFrame

print(df_filtered_sorted)

#Redraw boxplot but on filtered opposition list

sns.boxplot(x='opposition', y='runs_scored', data=df_filtered_sorted,palette= 'Paired')
plt.xticks(rotation=45)
plt.show()

#histogram (distplot) with and without kde (kernel density estimation )
#distplot = distribution plot and is same as 'histplot'

sns.displot(data = df_filtered_sorted, x = 'runs_scored', kde = False)
plt.show()

#we see that there is a right/positive skew, so there is

sns.displot(data = df_filtered_sorted, x = 'runs_scored', kde = True)
plt.show()

#histogram with bins

sns.set(style="darkgrid")
sns.histplot(data = df_new,x = 'runs_scored',bins = 15)
plt.show()


#KDE plot  ": will show the probability of data ranges

plt.figure(figsize=(12,8))

sns.kdeplot(data = df_new,x = 'runs_scored')
plt.show()


#KDE plot with cumulative probability

plt.figure(figsize=(12,8))

sns.kdeplot(data = df_new,x = 'runs_scored', cumulative=True)
plt.show()


#joint plot

sns.jointplot(x = 'balls_faced', y = 'runs_scored', data = df_new, kind = 'scatter')
plt.show()

#heat map
#calculate the correlation matrix

correlation_matrix = df_new[['balls_faced','runs_scored']].corr()

#Create the heatmap

plt.figure(figsize=(8,6))
sns.heatmap(data = correlation_matrix, annot =True, cmap = 'viridis', square = True, fmt =".2f")
plt.title('Correlation heatmap between balls faced and runs scored', fontsize = 14)
plt.show()

grouped_by_opposition = df_filtered.groupby('opposition')
#df1 = df_filtered.groupby('opposition')
agg_sum= grouped_by_opposition.agg({'runs_scored': 'sum','balls_faced':'sum'})
#calculate strike rate
df4 = agg_sum
df4['strike_rate']= (df4['runs_scored'] / df4['balls_faced'] ) * 100

plt.figure(figsize=(8,6))
sns.heatmap(data = df4,linewidths= 0.5,annot = True, cmap = 'viridis', square = True, fmt =".2f")
plt.show()

#plot a stack bar graph run_scored vs run_scored in boundaries
df_boundaries =  df_filtered[['opposition','runs_scored','fours','sixes']].copy()

df_boundaries['runs_scored_in_boundaries'] = (df_boundaries['fours'] * 4) + (df_boundaries['sixes'] * 6)

df_boundaries_selected = df_boundaries[['opposition','runs_scored','runs_scored_in_boundaries']].copy()
grouped_by_opposition = df_boundaries_selected.groupby('opposition')
df_boundaries_grouped = grouped_by_opposition.sum().reset_index()
print(df_boundaries_grouped)

#df_boundaries['runs_scored_in_boundary']= (grouped_by_opposition['fours'].sum() * 4) + (grouped_by_opposition['sixes'].sum() * 6)

#print("runs scored in boundary\n",df_boundaries)


#Stacked bar plot

plt.figure(figsize=(10,6))

plt.bar(df_boundaries_grouped['opposition'],df_boundaries_grouped['runs_scored'],label= 'Runs_Scored')

plt.bar(df_boundaries_grouped['opposition'],df_boundaries_grouped['runs_scored_in_boundaries'], bottom = df_boundaries_grouped['runs_scored'], label = 'Runs in Boundaries')

plt.xlabel('Opposition')
plt.ylabel('Runs')
plt.title('Runs Scored and Runs in Boundaries by Opposition')

plt.legend()

plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()
df_boundaries =  df_filtered[['opposition','runs_scored','fours','sixes']].copy()

df_boundaries['runs_scored_in_boundaries'] = (df_boundaries['fours'] * 4) + (df_boundaries['sixes'] * 6)

df_boundaries_selected = df_boundaries[['opposition','runs_scored','runs_scored_in_boundaries']].copy()
grouped_by_opposition = df_boundaries_selected.groupby('opposition')
df_boundaries_grouped = grouped_by_opposition.sum().reset_index()
print(df_boundaries_grouped)

#df_boundaries['runs_scored_in_boundary']= (grouped_by_opposition['fours'].sum() * 4) + (grouped_by_opposition['sixes'].sum() * 6)

#print("runs scored in boundary\n",df_boundaries)


#Stacked bar plot

plt.figure(figsize=(10,6))

plt.bar(df_boundaries_grouped['opposition'],df_boundaries_grouped['runs_scored'],label= 'Runs_Scored')

plt.bar(df_boundaries_grouped['opposition'],df_boundaries_grouped['runs_scored_in_boundaries'], bottom = df_boundaries_grouped['runs_scored'], label = 'Runs in Boundaries')

plt.xlabel('Opposition')
plt.ylabel('Runs')
plt.title('Runs Scored and Runs in Boundaries by Opposition')

plt.legend()

plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

