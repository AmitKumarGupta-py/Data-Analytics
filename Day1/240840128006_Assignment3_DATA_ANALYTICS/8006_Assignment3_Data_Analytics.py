import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.pylabtools import figsize
from mkl_random.mklrand import normal

print("------------------------------------------------------------------------")
print("Question 1")

df = pd.read_csv('Data Sets-20241014T055916Z-001\Data Sets\diamonds.csv')
print(df.head())

print("------------------------------------------------------------------------")
print("Question 2")

print(df.columns)

df['price_per_carat'] = df['price']/df['carat']

print(df.head(6))

column_set = ['index','carat', 'cut', 'color', 'clarity','depth','table','price','x_','y_','z_','price_per_carat',]
df2 = df.set_axis(column_set, axis=1)

print(df2.head(6))

print("------------------------------------------------------------------------")
print("Question 3")


print(df.color.unique())
print(df[df['color'] == 'E'].head())
print(df['cut'].head())

print("------------------------------------------------------------------------")
print("Question 4")

df['Quality_color'] = df['color']+ "_" + df['cut']

print(df.head(6))

print("------------------------------------------------------------------------")
print("Question 5")

print("No. of rows and columns: ",df.shape,"\n\n","dataType of each columns:\n", df.dtypes)

print("------------------------------------------------------------------------")
print("Question 6")

df2 = df.select_dtypes('object')
print(df2.head())

print("------------------------------------------------------------------------")
print("Question 7")

print(df.columns)

df.rename(columns={'x':'Length in mm','y':'width in mm','z':'depth in mm'},inplace=True)
print(df.head())
print(df.columns)

print("------------------------------------------------------------------------")
print("Question 8")

df.rename(columns={'carat':'Carat','cut':'Cut','color':'Color','clarity':'Clarity','depth':'Depth','table':'Table','price':'Price','x':'Length in mm','y':'width in mm','z':'depth in mm','price_per_carat':'Price_Per_Carat','Quality_color':'quality_color'},inplace=True)
print(df.columns)
print(df.head())

print("------------------------------------------------------------------------")
print("Question 9")

df.drop('Table',axis = 1,inplace=True)

print(df.head())


print("------------------------------------------------------------------------")
print("Question 10")

print(df.columns)
Columns = ['Clarity','quality_color']

df.drop(Columns,axis = 1,inplace=True)
print(df.columns)
print(df.head())


print("------------------------------------------------------------------------")
print("Question 11")

print(df.head(),"\n")
df_dropped = df.drop([0,2,5,8])
print(df_dropped.head())

print("------------------------------------------------------------------------")
print("Question 12")

df.sort_values(by='Cut',inplace=True)

print(df.head(10),"\n")


print("------------------------------------------------------------------------")
print("Question 13")

print(df.columns)
df.sort_values(by='Price',inplace=True,ascending=False)
print(df.head(10),"\n")


print("------------------------------------------------------------------------")
print("Question 14")

df.sort_values(by=['Carat'],inplace=True,ascending=False)
print(df.head(10),"\n")

df.sort_values(by=['Carat'],inplace=True,ascending=True)
print(df.head(10),"\n")


print("------------------------------------------------------------------------")
print("Question 15")

df_filtered = df[df['Carat'] >= 0.3]

print(df_filtered.head(10),"\n")


print("------------------------------------------------------------------------")
print("Question 16")

list1 = ['Apple','Bananas','Mango','Orange']
sr1 = pd.Series(list1)

print(sr1)

print("------------------------------------------------------------------------")
print("Question 17")

print(df.columns)
df_filtered = df[(df['Length in mm'] < 5)  & (df['width in mm'] < 5)  & df['depth in mm'] < 5]

print(df_filtered.head(10),"\n")

print("------------------------------------------------------------------------")
print("Question 18")

df_filtered  =df[(df['Cut'] == 'Premium') | (df['Cut'] == 'Ideal')]

print(df_filtered.head(10),"\n")

print("------------------------------------------------------------------------")
print("Question 19")

df_filtered  =df[(df['Cut'] == 'Good') | (df['Cut'] == 'Ideal') | (df['Cut'] == 'Fair')]

print(df_filtered.head(10),"\n")

print("------------------------------------------------------------------------")
print("Question 20")

print("Columns of DataFrame:\n",df.columns)


print("------------------------------------------------------------------------")
print("Question 21")

print(df.iloc[0:3],"\n\n")
print(df.head(3))

print("------------------------------------------------------------------------")
print("Question 22")

print("\nIterating using iterrows():")
for index, row in df.iterrows():
    print(f"Index: {index}, Cut: {row['Cut']}, Color: {row['Color']}, Carat: {row['Carat']}")

print("------------------------------------------------------------------------")
print("Question 23")

print(df.columns)

df1 = df.copy(deep=True)
df1.drop(df1.select_dtypes(include='object').columns,axis = 1,inplace=True)
print(df1.head(),"\n")

print(df1.info())



print("------------------------------------------------------------------------")
print("Question 24")


numeric_df = df.select_dtypes(include='number')
print(numeric_df.head(),"\n")


print("------------------------------------------------------------------------")
print("Question 25")

list1 = ['int64','float64']
numeric_df = df.select_dtypes(include=list1)
print(numeric_df.describe(),"\n")


print("------------------------------------------------------------------------")
print("Question 26")


df_mean = df.select_dtypes(include='number').mean()
print(df_mean)

print("------------------------------------------------------------------------")
print("Question 27")


numeric_df = df.select_dtypes(include='number')


row_means = numeric_df.mean(axis=1)


df['row_mean'] = row_means

print("\nDataFrame with row means added:")
print(df)

print("------------------------------------------------------------------------")
print("Question 28")

mean_price_per_cut = df.groupby('Cut')['Price'].mean()
print(df.columns)
print("\nMean price for each cut of diamonds:")
print(mean_price_per_cut)

print("------------------------------------------------------------------------")
print("Question 29")


summary_stats = df.groupby('Cut')['Price'].agg(['count', 'min', 'max']).reset_index()

print("\nCount, minimum, and maximum price for each cut of diamonds:")
print(summary_stats)


print("------------------------------------------------------------------------")
print("Question 30")




summary_stats = df.groupby(['Cut', 'Color'])['Price'].mean().unstack()


summary_stats.plot(kind='bar', figsize=(10, 6))


plt.title('Average Price of Diamonds by Cut and Color')
plt.xlabel('Cut')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.legend(title='Color')


plt.tight_layout()
plt.show()


print("------------------------------------------------------------------------")
print("Question 31")

df1 = df['Cut'].value_counts()
print(df1)


print("------------------------------------------------------------------------")
print("Question 32")

df1 = df['Cut'].value_counts(normalize = True) *100

print(df1)


print("------------------------------------------------------------------------")
print("Question 33")

df1 = df['Cut'].unique()
print(df1)

print("------------------------------------------------------------------------")
print("Question 34")

df1 = df['Cut'].nunique()
print(df1)


print("------------------------------------------------------------------------")
print("Question 35")

df1 = pd.crosstab(df['Cut'], df['Color'])

print(df1)

print("------------------------------------------------------------------------")
print("Question 36")

df1 = df['Cut'].describe()
print(df1)


print("------------------------------------------------------------------------")
print("Question 37")

plt.figure(figsize = (10,6))
#df['Carat'].hist(bins = 30, edgecolor = 'black')
sns.histplot(data=df, x='Carat', bins=35, kde=True)

plt.title('Histogram of Diamond Carats')
plt.xlabel('Carat')
plt.ylabel('Frequency')

plt.show()


print("--------------------------------------------------------------------------")
print("Question 38")

cut_counts = df['Cut'].value_counts()

plt.figure(figsize=(10,6))

sns.barplot(x=cut_counts.index , y=cut_counts.values, palette= 'viridis' )

plt.title('Count of Diamonds by Cut')
plt.xlabel('Cut')
plt.ylabel('Count')

plt.show()


print("--------------------------------------------------------------------------")
print("Question 39")

df1 = df.isnull()
print(df1)

print("--------------------------------------------------------------------------")
print("Question 40")


df1 = df.isnull().sum()
print(df1)


print("--------------------------------------------------------------------------")
print("Question 41")

df1 = df.shape

print(df1,"\n")


diamonds_cleaned = df.dropna()


num_rows_cleaned = diamonds_cleaned.shape[0]
print(f"Number of rows after dropping missing values: {num_rows_cleaned}")


print("--------------------------------------------------------------------------")
print("Question 42")

print("Original shape:", df.shape)

diamonds_any_missing = df.dropna(subset=['Carat', 'Price'], how='any')

print("Shape after dropping rows with any missing values:", diamonds_any_missing.shape)

diamonds_all_missing = df.dropna(subset=['Carat', 'Price'], how='all')

print("Shape after dropping rows with all missing values:", diamonds_all_missing.shape)


print("--------------------------------------------------------------------------")
print("Question 43")

df1 = df.reset_index()

print(df1)

df2 =df.set_index(keys = 'Carat')

print(df2)

print("--------------------------------------------------------------------------")
print("Question 44")

df3 = df.index
print(df3)

df1 =df.set_index(keys = 'Carat')
print(df1)

df2 = df.reset_index()

print(df2)


print("--------------------------------------------------------------------------")
print("Question 45")

cut_series = df['Cut']

print("Cut Series:")
print(cut_series)

specified_index = 10
value_at_index = cut_series.loc[specified_index]

print(f"\nValue at index {specified_index}: {value_at_index}")

positional_index = 10
value_at_positional_index = cut_series.iloc[positional_index]

print(f"\nValue at positional index {positional_index}: {value_at_positional_index}")

print("--------------------------------------------------------------------------")
print("Question 46")

df1 = df.sort_values(by='Price')
df2 = df.sort_index()

print("Sort by values:\n",df1)
print("\nSort by index:\n",df2)


print("--------------------------------------------------------------------------")
print("Question 47")

df['volume'] = df['Length in mm'] * df['width in mm'] * df['depth in mm']

print("\nDataFrame with Volume Column:")
print(df[['Cut', 'Carat', 'volume']].head())


print("--------------------------------------------------------------------------")
print("Question 48")


color_series = df['Color']

concatenated_df = pd.concat([df, color_series], axis=1)

concatenated_df.rename(columns={'Color': 'color_series'}, inplace=True)

print("\nConcatenated DataFrame:")
print(concatenated_df.head())

print("--------------------------------------------------------------------------")
print("Question 49")

print(df.iloc[4])

print("--------------------------------------------------------------------------")
print("Question 50")

print(df.iloc[[0,5,7]])


print("--------------------------------------------------------------------------")
print("Question 51")

print(df.iloc[2:6])


print("--------------------------------------------------------------------------")
print("Question 52")

df1 = df.loc[0:2 ,['Color','Price']]
print(df1)

print("--------------------------------------------------------------------------")
print("Question 53")

print(df.loc[0:3 ,'Color':'Price'])


print("--------------------------------------------------------------------------")
print("Question 54")

df1 = df.loc[df['Cut'] == 'Premium', 'Color']

print(df1)

print("--------------------------------------------------------------------------")
print("Question 55")

print(df.iloc[0:2][0:4])

print("--------------------------------------------------------------------------")
print("Question 56")


print(df.iloc[0:4][1:4])

print("--------------------------------------------------------------------------")
print("Question 57")

df1 = df.iloc[0:4, :]

print(df1)

print("--------------------------------------------------------------------------")
print("Question 58")

df1 = df.iloc[2:6, 0:2]

print(df1)

print("--------------------------------------------------------------------------")
print("Question 59")

print(df.head())
print(df.describe())
print(df.shape)
print(df.info)


print("--------------------------------------------------------------------------")
print("Question 60")

df1 = df.memory_usage(deep=True)
total_memory_usage = df1.sum()
print("Memory Usage of Each Column (in bytes):")
print(df1)

print("\nTotal Memory Usage of the DataFrame (in bytes):")
print(total_memory_usage)


print("--------------------------------------------------------------------------")
print("Question 61")

df1 = df.memory_usage(deep=True)
print("Memory Usage of Each Column (in bytes):")
print(df1)


print("--------------------------------------------------------------------------")
print("Question 62")

df1 = df.sample()
print(df1)

print("--------------------------------------------------------------------------")
print("Question 63")

sampled_df = df.sample(frac=0.75, random_state=1)


remaining_df = df.drop(sampled_df.index)

print("\nSampled DataFrame shape (75% of rows):", sampled_df.shape)
print("Remaining DataFrame shape (25% of rows):", remaining_df.shape)

print("--------------------------------------------------------------------------")
print("Question 64")

print("Original DataFrame:")
print(df.head())

duplicate_colors = df['Color'].duplicated(keep=False)

print("\nDuplicate Colors (True indicates duplicates):")
print(duplicate_colors)

duplicates_df = df[duplicate_colors]

print("\nRows with Duplicate Colors:")
print(duplicates_df)

print("--------------------------------------------------------------------------")
print("Question 65")

df1 = duplicates_df.duplicated().count()

print(df1)