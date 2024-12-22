import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

df = pd.read_csv('MS_Dhoni_ODI_record.csv')

#Basic checks

print(df.head())
print(df.tail())


#Data cleaning - Opposition  name says 'v Aus' etc,we can remove 'v '
#df['opposition'] = df['opposition'].apply(lambda x: x[2:])
df['opposition'] = df['opposition'].str.replace('v ','',regex=False)
#regex= False

#dropping the odi_number feature because it adds no value to the analysis

df.drop(columns='odi_number',inplace=True)

#dropping those innings where dhoni did not bat and stoding in new data frame
#Take all the columns, starting with run_scored

df_new = df.loc[((df['score'] != 'DNB') & (df['score'] != 'TDNB'))]



#Career stats
first_match_date = df['date'].df.date.min().strftime('%B %d, %Y')#first match %B = month %d= days %Y = year
print("First match",first_match_date)
last_match_date = df['date'].df.date.max().strftime('%B %d, %Y')
print("Last match",last_match_date)

number_of_matches = df.shape[0] #number of matches
print("Number of matches",number_of_matches)

number_of_inns = df_new.shape[0]# number of innings
print("Number of innings played",number_of_inns)

not_outs = df_new['not_out'].sum() #numbers of not outs in career
print("Total not outs",not_outs)

runs_scored = df_new['runs_scored'].sum() #runs scored in career
print("Total runs scored in career",runs_scored)
print('Balls faced in career:',balls_faced)
career_sr = (runs_scored/balls_faced)*100 #career strike rate
print('Career strike rate:{:.2f}'.format(career_sr))

career_avg = (runs_scored/(number_of_inns - not_outs)) #career average
print("Career average{:.2f}".format(career_avg))

#opposition_counts will be a series with a labelled index as opposition
opposition_counts = df_new['opposition'].value_counts()
print(opposition_counts)
#plot the counts as a bar plot
opposition_counts.plot(kind='bar',title='Number of matches against each opposition',figsize = (8,5))
plt.show()

#runs scored against each team
#Group the DataFrame by 'opposition' column
grouped_by_opposition = df_new.groupby('opposition')

#sum the 'runs_scored' column for each group
sum_of_runs_scored = grouped_by_opposition['runs_scored'].sum()
print(sum_of_runs_scored)

#sum_of_runs_scored is a series with a labelled index,which is opposition
#convert it into a Dataframe and remove  the index

runs_scored_by_opposition = pd.DataFrame(sum_of_runs_scored).reset_index()
runs_scored_by_opposition.plot(x='opposition',kind = 'bar',title = 'Runs scored against each opposition',figsize = (8,5))
plt.xlabel(None)
plt.show()