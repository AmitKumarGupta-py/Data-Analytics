import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from IPython.core.pylabtools import figsize
from bokeh.plotting import figure
from pandas import value_counts

print("------------------------------------------------------------------")
print("Ques 1")
print("Code 1.1")
data1 = {
    'Tom': [23, 'John', 45.5, 78, 'Alice', 67, 'Bob', 34, 90.1, 'Charlie'],
    'Brick': [True, False, 'Blue', 'Red', 12, 'Green', False, 24.5, 'Yellow', 10],
    'Harry': [None, 'Hello', 3.14, 42, 'World', True, 'Python', np.nan, 100, 'Test']
}

df1 = pd.DataFrame(data1, index =(True,False) * 5)
print("\nData1:\n",df1)

df1.to_csv("Data1.csv",index = True)

data2 = {
    'Tom': [99, 'Emily', 28.4, 45, 'David', 55, 'Eve', 67, 'Gina', 80],
    'Brick': [False, 'Orange', 15, True, 'Pink', 'Brown', 30.2, True, 'Black', 5],
    'Harry': [5, 'Test1', None, 88, 'Example', 12.34, False, 'Sample', np.nan, 'Final']
}


df2 = pd.DataFrame(data2, index = (True,False) * 5)
print("\nData2:\n",df2)

print("\nCode 1.2")


data = {
    'A': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
    'B': ['apple', 'banana', np.nan, 'date', 'fig', 'grape', np.nan, 'kiwi', 'lemon', 'mango'],
    'C': [10, np.nan, 30, 40, 50, 60, 70, np.nan, 90, 100],
    'D': [True, False, True, False, True, False, True, True, False, True],
    'E': [np.nan, 'red', 'blue', 'green', np.nan, 'yellow', 'purple', 'orange', np.nan, 'pink']
}

df =pd.DataFrame(data)
print("\nData:\n",df)

df['A'].fillna(df['A'].mean(),inplace = True)
df['C'].fillna(df['C'].mean(),inplace = True)

print("\npartially Cleaned Data1:\n",df)

for index, row in df.iterrows():
    if pd.isnull(row['B']) or pd.isnull(row['E']):
        df.drop(index, inplace=True)

df.reset_index(drop = True, inplace= True)

print("\nCleaned data:\n",df)

print("\nCode 1.3\n")

data = {
    'Abilash': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'Ankit': [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
    'Ashok': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
    'Asif': [2, 12, 22, 32, 42, 52, 62, 72, 82, 92],
    'Anjaan': [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
}

df = pd.DataFrame(data)

print("\nData:\n",df)

group_mapping = {
    10: 'Group 1',
    20: 'Group 1',
    30: 'Group 2',
    40: 'Group 2',
    50: 'Group 3',
    60: 'Group 3',
    70: 'Group 4',
    80: 'Group 4',
    90: 'Group 5',
    100: 'Group 5'
}


df['Group'] = df['Abilash'].map(group_mapping)

grouped_df = df.groupby('Group').sum()

print("\nGrouped DataFrame using a mapper:")
print(grouped_df)

df['Ankit_Range'] = df['Ankit'] // 20
print("\n#########\n",df)

grouped_multiple_df = df.groupby(['Group', 'Ankit_Range']).sum()

print("\nGrouped DataFrame by multiple columns:")
print(grouped_multiple_df)


print("\nCode 1.4\n")

df = pd.read_csv("CountryDataIND.csv")
print(df.columns)
print(df.head())
print(df.info())

print("\nMissing data check:\n",df.isnull().sum())

df['Footnotes'].fillna(0,inplace = True)
print("\nFillna operation:\n",df.head(10))

print("\n!!!!!!!!!!\n",df.isna().head(10))

print("\nCode 1.5\n")

plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.hist(df['Observation Value'],bins = range(0,101,10),edgecolor = 'black')
plt.title('Histogram of Observation Value')
plt.xlabel('Observation Value')
plt.ylabel('Frequency')
plt.xticks(range(0, 101, 10))

plt.subplot(1, 2, 2)
plt.scatter(df['Time period details'], df['Observation Value'], color='blue')
plt.title('Scatter Plot of Observation Value vs Time Period')
plt.xlabel('Time Period')
plt.ylabel('Observation Value')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


print("------------------------------------------------------------------")
print("Ques 2")
print("Code 2.1")

df1 = pd.read_csv("purchase_data.csv")
print(df1.columns)
print(df1.head())
print(df1.info())

print("\nTotal Number of Players:\n",df1['SN'].count())

print("Code 2.2")
print("\nPurchasing Analysis (Total)\n")
print("\nNumber of Unique Items:\n",df1['Item ID'].nunique())
print("\nAverage Purchase Price $:\n",df1['Price'].mean())
print("\nTotal Number of Purchases:\n",df1['Purchase ID'].count())
print("\nTotal Revenue $:\n",df1['Price'].sum())


print("\nCode 2.3\n")
print("\nGender Demographics\n")
df_grouped = df1.groupby('Gender')['Gender'].value_counts()

print("\nGrouped Data:\n",df_grouped)

#percentage_male = (df_grouped['Male'] / (df_grouped['Male'] + df_grouped['Female'] + df_grouped['Other / Non-Disclosed'])) * 100
unique_male = df1[df1['Gender'] == 'Male']['SN'].nunique()
print("\nCount of Male Players:\n",unique_male)

#percentage_Female = (df_grouped['Female'] / (df_grouped['Male'] + df_grouped['Female'] + df_grouped['Other / Non-Disclosed'])) * 100
unique_Female = df1[df1['Gender'] == 'Female']['SN'].nunique()
print("\nCount of Female Players:\n",unique_Female)
#print("\nPercentage and Count of Female Players\n",percentage_Female)

percentage_Other = (df_grouped['Other / Non-Disclosed'] / (df_grouped['Male'] + df_grouped['Female'] + df_grouped['Other / Non-Disclosed'])) * 100

unique_Other = df1[df1['Gender'] == 'Other / Non-Disclosed']['SN'].nunique()
print("\nCount of Other Players:\n",unique_Other)
print("\nPercentage and Count of Other / Non-Disclosed Players\n",percentage_Other)
percentage_male = (unique_male / (unique_male + unique_Female + unique_Other)) * 100
print("\nPercentage of Male Players:\n",percentage_male)

percentage_female = (unique_Female / (unique_male + unique_Female + unique_Other)) * 100
print("\nPercentage of feMale Players:\n",percentage_female)

percentage_Other = (unique_Other / (unique_male + unique_Female + unique_Other)) * 100
print("\nPercentage of Other Players:\n",percentage_Other)


print("\nCode 2.4\n")
print("\nPurchasing Analysis (Gender)\n")

df_grouped_purchased = df1.groupby('Gender')['Purchase ID'].count()
print("\nPurchase Count:\n",df_grouped_purchased)

print("\nAverage Purchase Price $(Gender)\n")
df_average_price = df1.groupby('Gender')['Price'].mean()
print("\nAverage Purchase Total per Person by Gender:\n",df_average_price)
df_avg = df1['Price'].mean()
print("\nTotal Average$:\n",df_avg)

df_total = df1.groupby('Gender')['Price'].sum()
print("\nTotal Purchase Value$ by gender\n",df_total)

df_sum = df1['Price'].sum()
print("\nTotal Purchase value$:\n",df_sum)


print("\nCode 2.5\n")
print("\nAge Demographics...\n")

bins = [-1, 4, 8, 12, 16, 20,24]
labels = ['<10', '10-14', '15-19', '20-24', '25-29','30-34']

df1['Age Group'] = pd.cut(df1['Age'], bins=bins, labels=labels, right=True)

age_demographics = df1['Age Group'].value_counts().sort_index()

print("Age Group",df1)

age_demographics = df1.groupby('Age Group').agg(
    Purchase_Count=('Price', 'size'),
    Average_Purchase_Price=('Price', 'mean'),
    Total_Purchase_Value=('Price', 'sum'),
    Average_Purchase_Total_Per_Person=('Price', lambda x: x.sum() / x.count())
)

print("Age Demographics:")
print(age_demographics)

print("\nPurchase_Count:\n",age_demographics['Purchase_Count'])
print("\nAverage_Purchase_Price:\n",age_demographics['Average_Purchase_Price'])
print("\nTotal_Purchase_Value:\n",age_demographics['Total_Purchase_Value'])
print("\n Average_Purchase_Total_Per_Person:\n",age_demographics['Average_Purchase_Total_Per_Person'])

print("\nCode 2.6\n")
print("\nTop Spenders\n")

spender_df = df1.groupby('SN')['Price'].sum().reset_index()
top_spender = spender_df.sort_values(by='Price', ascending=False)
print("\nthe top 5 spenders in the game by total purchase value $:\n",top_spender.head())

spender_analysis = df1.groupby('SN').agg(
    Purchase_Count=('Price', 'size'),
    Average_Purchase_Price=('Price', 'mean'),
    Total_Purchase_Value=('Price', 'sum')
).reset_index()


spender_analysis = spender_analysis.sort_values(by='Total_Purchase_Value', ascending=False)


top_5_spenders = spender_analysis.head(5)

top_5_spenders.insert(0, 'Name', range(1, len(top_5_spenders) + 1))

print("Top 5 Spenders:")
print(top_5_spenders.head())

print("\nCode 2.7\n")
print("\nMost Popular Items\n")

print(df1.columns)
Item_count = df1['Item ID'].nunique()
print(Item_count)
df1['Item_count'] = df1['Item ID'].value_counts()
print(df1.head())

df1['Total Purchase Value'] = df1['Item_count'] * df1['Price']

print("\nCode 2.8\n")
print("\nMost Profitable Items:\n")

top_items = df1.sort_values(by='Total Purchase Value', ascending=False).head(5)
print(top_items.head())
print(top_items.columns)
result = top_items[['Purchase ID','SN', 'Item ID','Item Name', 'Item_count', 'Price', 'Total Purchase Value']]

print(result)



print("------------------------------------------------------------------")
print("Ques 3")

df2 = pd.read_csv("clean_18.csv")
print(df2.head())
print(df2.columns)

print("------------------------------------------------------------------")
print("Ques 4")

df3 = pd.read_csv("all_alpha_18.csv")
print(df3.columns)

print("\nData Cleaning:\n")

print(df.describe())

missing_values = df3.isnull().sum()
print("\nmissing_values:\n",missing_values[missing_values > 0])

df3['Displ'] = df3['Displ'].ffill()
df3['Cyl'] = df3['Cyl'].bfill()

print(df3[['Displ','Cyl']].head())

uniques1 = df3.nunique()
print('\nUniques:\n',uniques1)

unique_rows = df3.drop_duplicates()
print("\nDataFrame with Unique Rows:")
print(unique_rows)

print("\nRenaming Columns:\n")
df3.rename(columns={'Cyl': 'Cylinder', 'Cert Region': 'Region'}, inplace=True)
print("\nRenamed DataFrame Columns:")
print(df3.columns)

print(df3.describe())

print(df3.info())


df3['Cylinder'] = df3['Cylinder'].astype(int)
df3['City MPG'] = pd.to_numeric(df3['City MPG'], errors='coerce')
df3['Hwy MPG'] = pd.to_numeric(df3['Hwy MPG'], errors='coerce')
df3['Cmb MPG'] = pd.to_numeric(df3['Cmb MPG'], errors='coerce')

df3['Fuel'] = df3['Fuel'].str.lower().str.strip()
print(df3['Fuel'].head(10))


sns.boxplot(x=df3['Air Pollution Score'])
plt.show()

sns.boxplot(x=df3['City MPG'])
plt.show()
################################################################
mean_scores = df3.groupby('Veh Class')['Air Pollution Score'].mean().reset_index()

mean_scores.sort_values(by='Air Pollution Score', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Air Pollution Score', y='Veh Class', data=mean_scores, palette='viridis')

plt.title('Average Air Pollution Score by Vehicle Class', fontsize=16)
plt.xlabel('Average Air Pollution Score', fontsize=14)
plt.ylabel('Vehicle Class', fontsize=14)

plt.show()

###########################################################
mean_scores = df3.groupby('Veh Class')['Greenhouse Gas Score'].mean().reset_index()

mean_scores.sort_values(by='Greenhouse Gas Score', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Greenhouse Gas Score', y='Veh Class', data=mean_scores, palette='viridis')

plt.title('Average Greenhouse Gas Score by Vehicle Class', fontsize=16)
plt.xlabel('Average Greenhouse Gas Score', fontsize=14)
plt.ylabel('Vehicle Class', fontsize=14)

plt.show()

############################################################
print("Ques 4.1")
print("\nThe alternative sources of fuel available in 2008 & 2018 respectively and by how much?\n")

df5 = pd.read_csv("clean_18.csv")
df6 = pd.read_csv("clean_08.csv")


df5['fuel'] = [row.lower() for row in df5['fuel']]
df6['fuel'] = [row.lower() for row in df6['fuel']]

print("\nFuel Sources in 2008:\n",df5['fuel'].value_counts())
print("\nFuel Sources in 2018:\n",df6['fuel'].value_counts())

print(df5.columns,"\n@@@@@@@@@@@@@@@@@@\n",df6.columns)

print("\nMpg 2008\n",df5['cmb_mpg'].value_counts().mean())
print("\nMpg 2018\n",df6['cmb_mpg'].value_counts().mean())


print("\nChange in characteristics of SmartWay Vehicles:\n")


print("\nSmart way vehicle in 2008:\n",df5['smartway'].value_counts())
print("\nSmart way vehicle in 2018:\n",df6['smartway'].value_counts())

numeric_df = df5.select_dtypes(include=[float, int])
ap = numeric_df.corr()

numeric_df = df6.select_dtypes(include=[float, int])
dp = numeric_df.corr()


sns.heatmap(ap, annot=True,cmap='coolwarm',linewidths=0.5,linecolor='black')
plt.show()

sns.heatmap(dp, annot=True,cmap='coolwarm',linewidths=0.5,linecolor='black')
plt.show()



print("------------------------------------------------------------------")
print("Ques 5")

df9C = pd.read_csv("LS2009Candidate.csv")
df14C = pd.read_csv("LS2014Candidate.csv")
df9E = pd.read_csv("LS2009Electors.csv")
df14E = pd.read_csv("LS2014Electors.csv")

print("\n2009C:\n",df9C.info())
print("\n2014C:\n",df14C.info())
print("\n2009E:\n",df9E.info())
print("\n2014E:\n",df14E.info())


UPA = ['INC','NCP', 'RJD', 'DMK', 'IUML',
'JMM','JD(s)','KC(M)','RLD','RSP','CMP(J)','KC(J)','PPI','MD']

NDA = ['BJP','SS', 'LJP', 'SAD', 'RLSP',
'AD','PMK','NPP','AINRC','NPF','RPI(A)','BPF','JD(U)','SDF','NDPP','MNF','RIDALOS','KMDK','IJK','PNK','JSP','GJM','MGP','GFP','GVP','AJSU','IPFT','MPP','KPP','JKPC','KC(T)','BDJS','AGP','JSS','PPA','UDP','HSPDP','PSP','JRS','KVC','PNP','SBSP','KC(N)','PDF','MDPF']

Others = ['YSRCP','AAAP', 'IND', 'AIUDF', 'BLSP', 'JKPDP', 'JD(S)', 'INLD', 'CPI', 'AIMIM',
'KEC(M)','SWP', 'NPEP', 'JKN', 'AIFB', 'MUL', 'AUDF', 'BOPF', 'BVA', 'HJCBL',
'JVM','MDMK']


df9C['Alliance'] = ''

for index, row in df9C.iterrows():
    if row['Party Abbreviation'] in UPA:
        df9C.at[index, 'Alliance'] = 'UPA'

for index, row in df9C.iterrows():
    if row['Party Abbreviation'] in NDA:
        df9C.at[index, 'Alliance'] = 'NDA'

for index, row in df9C.iterrows():
    if row['Party Abbreviation'] in Others:
        df9C.at[index, 'Alliance'] = 'Others'

df9C.groupby('Alliance')['Alliance'].sum()

print(df9C.columns)


df14C['Alliance'] = ''

for index, row in df14C.iterrows():
    if row['Party Abbreviation'] in UPA:
        df14C.at[index, 'Alliance'] = 'UPA'

for index, row in df14C.iterrows():
    if row['Party Abbreviation'] in NDA:
        df14C.at[index, 'Alliance'] = 'NDA'

for index, row in df14C.iterrows():
    if row['Party Abbreviation'] in Others:
        df14C.at[index, 'Alliance'] = 'Others'

df14C.groupby('Alliance')['Alliance'].sum()

print(df14C.columns)


def winning_seats_distribution(df9C):
    # Filter for winners (assuming Position 1 indicates a win)
    winners = df9C[df9C['Position'] == 1]

    # Count winning seats by party and year
    distribution = winners.groupby(['Year', 'Party Abbreviation']).size().reset_index(name='Winning Seats')

    return distribution


# Get distributions for both years
distribution_2009 = winning_seats_distribution(df9C)
distribution_2014 = winning_seats_distribution(df14C)

# Combine distributions
combined_distribution = pd.concat([distribution_2009, distribution_2014])

# Display the combined winning seats distribution
print(combined_distribution)

# combined_distribution.reset_index()
# print(combined_distribution)

# Filter for winners
winners_2009 = df9C[df9C['Position'] == 1]

# Group by candidate category and count winning seats
seats_distribution = winners_2009['Candidate Category'].value_counts().reset_index()
seats_distribution.columns = ['Candidate Category', 'Winning Seats']

# Plotting
plt.figure(figsize=(8, 5))
plt.bar(seats_distribution['Candidate Category'], seats_distribution['Winning Seats'], color=['blue', 'orange', 'green'])
plt.title('Seats Won by Candidate Category in 2009')
plt.xlabel('Candidate Category')
plt.ylabel('Number of Winning Seats')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


winners_2014 = df14C[df14C['Position'] == 1]

# Group by candidate category and count winning seats
seats_distribution14 = winners_2014['Candidate Category'].value_counts().reset_index()
seats_distribution14.columns = ['Candidate Category', 'Winning Seats']

# Plotting
plt.figure(figsize=(8, 5))
plt.bar(seats_distribution['Candidate Category'], seats_distribution14['Winning Seats'], color=['blue', 'orange', 'green'])
plt.title('Seats Won by Candidate Category in 2014')
plt.xlabel('Candidate Category')
plt.ylabel('Number of Winning Seats')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


print("----------------------- Age Distribution ------------------------------")

winners_2009 = df9C[df9C['Position'] == 1]
winners_2014 = df14C[df14C['Position'] == 1]

# Combine age data for both years
ages = pd.concat([winners_2009['Candidate Age'], winners_2014['Candidate Age']])

# Plotting age distribution
plt.figure(figsize=(10, 6))
plt.hist(ages, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Age Distribution of Winners (2009 & 2014 Elections)')
plt.xlabel('Candidate Age')
plt.ylabel('Number of Winners')
plt.grid(axis='y')
plt.xticks(range(20, 70, 5))
plt.show()

print("-------------------Gender Distribution---------------------------")


winners_2009 = df9C[df9C['Position'] == 1]
winners_2014 = df14C[df14C['Position'] == 1]

# Count gender distribution for both years
gender_distribution_2009 = winners_2009['Candidate Sex'].value_counts()
gender_distribution_2014 = winners_2014['Candidate Sex'].value_counts()

# Combine the gender distributions into a single DataFrame
gender_distribution = pd.DataFrame({
    '2009': gender_distribution_2009,
    '2014': gender_distribution_2014
}).fillna(0)  # Fill NaN values with 0

# Plotting
gender_distribution.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'lightgreen'])
plt.title('Gender Distribution of Winners (2009 & 2014 Elections)')
plt.xlabel('Gender')
plt.ylabel('Number of Winners')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.legend(title='Year')
plt.show()

print("-------------------Poll Percentage---------------------------")

df9E.set_index('STATE', inplace=True)
df9E.groupby('STATE')

# Plotting
df9E.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'lightgreen'])
plt.title('Poll Percentage by State (2009 )')
plt.xlabel('State name')
plt.ylabel('Poll Percentage')
plt.xticks(rotation=45)
plt.ylim(0, 100)  # Setting y-axis limit to 100%
plt.grid(axis='y')
plt.legend(title='Year')
plt.show()