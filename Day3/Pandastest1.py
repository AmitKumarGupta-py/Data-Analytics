import numpy as np
import pandas as pd
from zope.interface import providedBy

np.random.seed(101)
mydata = np.random.randint(0,101,(4,3))
print(mydata)

myindex = ['CA','NY','AZ','TX']
mycolumns = ['JAN','FEB','MAR']

df = pd.DataFrame(data = mydata)
print(df)

df = pd.DataFrame(data = mydata,index = myindex)
print(df)

df = pd.DataFrame(data = mydata,index = myindex,columns = mycolumns)
print(df)

print(df.info())

df = pd.read_csv('tips.csv')
print(df)

print(df.columns)

print(df.index)

print(df.head(3))

print(df.tail(3))

print(df.info())

print("length: ",len(df))

print(df.describe())

print(df.describe().transpose())#statistical summary better organised


#select a single column ... first name
print(df['total_bill']) #print(df['first name']
print(df.total_bill) #print(df.first name)  #ERROR!!!

print(type(df['total_bill']))

print(df[['total_bill','tip']])

#create new column
df['tip_percentage'] = 100 * df['tip']/df['total_bill']
print(df.head())

df['price_per_person'] = df['total_bill']/df['size']
print(df.head())

#adjust existing columns
df['price_per_person'] = np.round(df['price_per_person'],2)
print(df.head())

df = df.drop("tip_percentage",axis = 1)
print(df.head())

print(df.index)

df = df.set_index('Payment ID')

print(df.head())

df = df.reset_index()

print(df.head())

print(df['total_bill'] < 30) #True / false

bool_series = df ['total_bill'] > 30 #save in variable

print(bool_series)
print(df[bool_series])#actual results

print(df[df['total_bill'] > 30])

#another syntax

print(df.total_bill > 30)
print(df[df.total_bill > 30])
print(df[df['gender'] == 'Male'])

#multiple conditions

df_new = df[(df['total_bill'] >30 ) & (df['gender'] == 'Male')]

print(df_new)

df_new = df[(df['total_bill'] >30 ) &  ~(df['gender'] == 'Male')]
print(df_new)

df_new = df[(df['total_bill'] >30 ) & (df['gender'] != 'Male')]
print(df_new)


df_new = df[(df['total_bill'] >30 ) | (df['tip'] > 5)]
print(df_new)

#Conditional isin operator
print("DF!!!!!!!!!!\n",df.info())
options = ['Sat','Sun']
print(df['day'].isin(options))
print(df[df['day'].isin(['Sat','Sun'])].day)

print(df[df['day'].isin(['Sat','Sun'])])

#sort values
print(df.sort_values('tip'))

#correlation
print(df[['total_bill','tip']].corr())

#idxmin and idxmax

print(df['total_bill'].max())
print(df['total_bill'].idxmax())
print(df['total_bill'].idxmin())

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(df.iloc[67])
print(df.iloc[170])

#value_counts - get count on categorical columns

print(df['smoker'].value_counts()) #smokers and non smokers
print(df['gender'].value_counts()) #males and females

#unique

print(df['size'].unique()) #all the uniques= values as an array
print(df['size'].nunique())#number of unique values (i.e count of the above array)
print(df['time'].unique())#FOR A char/string column

#b/w options for inclusive are both left right , neither
print("================================")
print(df['total_bill'].between(10,20,inclusive = 'both'))#True/false
print(df[df['total_bill'].between(10,20,inclusive = 'both')]) #actual data

#sample data randomly selected

print(df.sample(5))#5 rows
print(df.sample(frac = 0.1))#10% of rows

#nlargest and n smallest

print(df.nlargest(10,'tip'))
print(df.nsmallest(10,'tip'))

#groupby Functionality
#1.Splitting the data is first split into groups based on criteria provided
#2. Applying: A function (or multiple functions) is applied to each group independently
#3: Combining :can combine into a new dataFrame

#1:Splitting
df_grouped =   df.groupby('gender')

#note that the output will not have any visible impact,except for
#a mention of grouping
print(df_grouped)

#2. Applying
#Calculate the mean tip for each gender

mean_tip_by_gender = df_grouped['tip'].mean()
print(mean_tip_by_gender)

print(df_grouped['tip'].min())
print(df_grouped['tip'].max())
print(df_grouped['tip'].std())
print(df_grouped['tip'].var())
print(df_grouped['tip'].count())
print(df_grouped['tip'].sum())


#Creating groups on multiple columns

df_grouped = df.groupby(['gender','day'])
print(df_grouped['tip'].mean())



#agg function
#Apply multiple aggregation functions to different columns when grouping data
#calculate both the mean and sum of the tip column for each combination of 'gender and 'day'

#3:Combine
#group the DAtaframe by both 'gender  and day' and calculate the mean and  sum of tip each combination
print("----------------------------------------------")
df_grouped = df.groupby(['gender','day']).agg({'tip':['mean','sum']}).reset_index()
#reset index according to grouped columns
print(df_grouped)

#Rename the columns for clarity
print("############################################")
df_grouped.columns = ['gender','day','mean_tip','total_tip']
print(df_grouped)