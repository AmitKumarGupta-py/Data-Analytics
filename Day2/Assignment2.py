import pandas as pd
import numpy as np
from collections import Counter


print("\nQues1:\n")
arr = np.array([1,2,3,4,5,6,7,8,9])
print(arr)
df = pd.DataFrame(arr)
print(df)

print("--------------------------------------------------------------------")
print("\nQues2:\n")
df1 = pd.Series([1,2,3,4,5,6,7,8,9])
print(df1)

print("--------------------------------------")
df2 = df1.tolist()
print(df2,"\n",type(df2))

print("-------------------------------------------------------")
print("\nQues3:\n")
pd_Series1 = pd.Series([2,4,6,8,10])
pd_Series2 = pd.Series([1,3,5,7,9])


#add_Series = pd_Series1 + pd_Series2
add_Series = pd_Series1.add(pd_Series2)
sub_Series = pd_Series1 - pd_Series2
mul_Series = pd_Series1 * pd_Series2
div_Series = pd_Series1 / pd_Series2
print("------------------------------------")
print("Series Addition \n",add_Series)
print("Series Subtraction\n ",sub_Series)
print("Series Multiplication\n ",mul_Series)
print("Series Division\n ",div_Series)


print("-------------------------------------------------------")
print("\nQues4:\n")
s1 = pd.Series([2,4,6,8,10])
s2 = pd.Series([1,3,5,7,10])

c = s1.compare(s2)

print(c)

c1 = s1>s2
c2 = s1<s2
c3 = s1==s2
c4 = s1!=s2

print("Greater than\n",c1,"\n")
print("Less than\n",c2,"\n")
print("Equal to\n",c3,"\n")
print("Not equal to\n",c4,"\n")

print("-------------------------------------------------------")
print("\nQues5:\n")

Original_dictionary = {'a':100,'b':200,'c':300,'d':400,'e':800}
print("Dictionary before conversion into series:\n",Original_dictionary)
df3 = pd.Series(Original_dictionary)
print("Dictionary after conversion into series:\n",df3)

print("-------------------------------------------------------")
print("\nQues6:\n")

array1 = [10,20,30,40,50]
Numpy_array = np.array(array1)
print("Numpy array before conversion into series:\n",Numpy_array)
df4 = pd.Series(Numpy_array)
print("Numpy array after conversion into series:\n",df4)

print("-------------------------------------------------------")
print("\nQues7:\n")

sample = [100,200,'python',300.12,400]
s1 = pd.Series(sample)

print("Series:\n",s1)

s2 = pd.to_numeric(s1, errors='coerce',downcast='float')

print("\nAfter conversion into float:\n",s2)

print("-------------------------------------------------------")
print("\nQues8:\n")

data = [[1,4,7],[2,5,5],[3,6,8],[4,9,12],[7,5,1],[11,0,11]]
df = pd.DataFrame(data,columns=['col1','col2','col3'])
print("DataFrame:\n",df)

s1 = pd.Series(df['col1'])
print("Series:\n",s1)


print("-------------------------------------------------------")
print("\nQues9:\n")

s1 = pd.Series([100,200,'python',300.12,400])
print("Series:\n",s1)

arr = pd.array(s1, copy=True)
print("\nPandas array:\n",arr,"\nType: ",type(arr))
arr1 = np.array(arr, copy=True)
print("\nNumpy array:\n",arr1,"\nType: ",type(arr1))

print("-------------------------------------------------------")
print("\nQues10:\n")

s1 = pd.Series([['Red','Green','White'],['Red','Black'],['Yellow']])

print("Original Series:\n",s1)

s2 = s1.explode().reset_index(drop=True)
print("\nONE Series:\n",s2)

print("-------------------------------------------------------")
print("\nQues11:\n")

s1 = pd.Series(['100','200','python','300.12','400'])
print("Series:\n",s1)

s2 = s1.sort_values().reset_index(drop=True)
print("\nSorted Series:\n",s2)
s1.sort_values()
print(s1)

print("----------------------------------------------------------")
print("\nQues12:\n")

s1 = pd.Series([100,200,'python',300.12,400])
print("Series:\n",s1)
s2 = s1._append(pd.Series([500,'php']),ignore_index=True)

print("Data Series after adding some data: \n",s2,"\n")

print("----------------------------------------------------------")
print("\nQues13:\n")

s1 = pd.Series([0,1,2,3,4,5,6,7,8,9,10])
print("Original Series:\n",s1)

print("Subset of the above data Series:\n",s1[0:6])

print("----------------------------------------------------------")
print("\nQues14:\n")

s1 = pd.Series([1,2,3,4,5],index =['A','B','C','D','E'])

print("Original Data Series:\n",s1)

new_index_order = ['B', 'A', 'C', 'D', 'E']
s2 = s1.reindex(new_index_order)

print("\nData Series after re-indexing: \n",s2)

print("----------------------------------------------------------")
print("\nQues15:\n")

s1 = pd.Series([1,2,3,4,5,6,7,8,9,5,3])

print("Original Data Series:\n",s1)

print("Mean of the above data Series:\n",s1.mean())
print("Standard Deviation of the above data Series:\n",s1.std())

print("----------------------------------------------------------")
print("\nQues16:\n")

sr1 = pd.Series([1,2,3,4,5])
sr2 = pd.Series([2,4,6,8,10])

print("Original Data Series:\n",sr1,"\n",sr2)

sr3 = ~(sr1.isin(sr2))

print("\nItems of sr1 not present in sr2:\n",sr1[sr3])

print("----------------------------------------------------------")
print("\nQues17:\n")

sr1 = pd.Series([1,2,3,4,5])
sr2 = pd.Series([2,4,6,8,10])

print("Original Data Series:\n",sr1,"\n",sr2)

sr3 = sr1[~(sr1.isin(sr2))]
sr4 = sr2[~(sr2.isin(sr1))]

sr5 = pd.concat([sr3,sr4])


print("\nItems not common in both series:\n",sr5)

print("----------------------------------------------------------")
print("\nQues18:\n")

data = np.arange(3,22)
print("Data:\n",data)

sr1 = pd.Series(data)
print("Original Series:\n",sr1)

minimum  = sr1.min()
percentile_25 = sr1.quantile(0.25)
median1 = sr1.median()
percentile_75 = sr1.quantile(0.75)
maximum = sr1.max()

print("\nMinimum Value:\n",minimum,"\n")
print("25th Percentile Value:\n",percentile_25,"\n")
print("Median Value:\n",median1,"\n")
print("75th Percentile Value:\n",percentile_75,"\n")
print("Maximum Value:\n",maximum,"\n")


print("----------------------------------------------------------")
print("\nQues19:\n")

sr1 = pd.Series([1,7,1,6,3,4,5,6,7,5,8,9,10,8,7,6,5,4,3,2,1,11,12,13,14,11,12,12,1,1,20,0,0])

print("Original Series:\n",sr1)

print("\nFrequency of each unique Value of the said series:\n",sr1.value_counts())

print("-------------------------------------------------------------")
print("\nQues20:\n")

sr1 = pd.Series([1,7,1,6,3,3,3,3,4,5,6,7,5,8,9,10,8,7,6,5,4,3,2,1,11,12,13,14,11,12,12,1,1,20,0,0])

print("Original Series:\n",sr1)

# Get the frequency counts
value_counts = sr1.value_counts()

# Determine the maximum frequency
max_frequency = value_counts.max()

# Get all numbers with the maximum frequency
most_frequent_values = value_counts[value_counts == max_frequency].index

# Replace all other values with 'other'
sr3 = sr1.where(sr1.isin(most_frequent_values), 'other')

print("Series after modification:\n", sr3)

'''sr2 = sr1.value_counts().idxmax()


sr3 = sr1.where(sr1 == sr2,'other')

print("Series after modification:\n",sr3)'''

print("-------------------------------------------------------------")
print("\nQues21:\n")

sr1 = pd.Series([1,9,8,60,9,75,1,1,1])
pos = sr1[sr1 % 5 == 0].index.tolist()

print("the positions of numbers that are multiples of 5 in given Series:\n",pos)

print("-------------------------------------------------------------")
print("\nQues22:\n")

sr1 = pd.Series([2,3,9,0,2,3,4,5,3,2,0,4,5,2,3,4,0,5,9,4,2,4,5,0,3,4,6,7,3,7,9,3,6,8,3,7,8,4,7,9,5,9,7])

print("Original Series:\n",sr1)
pos = (0,2,6,11,21)
print("\nExtract items at given positions of the said series:\n",sr1.take(pos))

print("-------------------------------------------------------------")
print("\nQues23:\n")

sr1 = pd.Series([1,2,3,4,5,6,7,8,9,10])
sr2 = pd.Series([1,3,5,7,10])

print("First Series:\n",sr1)
print("Second Series:\n",sr2)

sr3 = sr1.isin(sr2)

print("\nPositions of items of series2 in series1:\n",sr1[sr3].index)

print("-------------------------------------------------------------")
print("\nQues24:\n")

sr1 = pd.Series(['php','python','java','c#'])
print("Original Series:\n",sr1)

'''def convert_case(sr1):
    if len(sr1) > 1:
        return sr1[0].upper() + sr1[1:-1] + sr1[-1].upper()
    elif len(sr1) == 1:
        return sr1.upper()
    return sr1'''

sr2 = sr1.apply(lambda x: x[0].upper() + x[1:-1] + x[-1].upper())

print("\nSeries after modification:\n",sr2)

# Applying the function to the Series
#modified_series = sr1.apply(convert_case)

# Display the result
#print(modified_series)


print("---------------------------------------------------")
print("\nQues25:\n")

sr1 = pd.Series(['php','python','java','c#'])
print("Original Series:\n",sr1)

sr2 = sr1.apply(lambda x:len(x))

print("\nSeries :\n",sr2)

print("---------------------------------------------------")
print("\nQues26:\n")

sr1 = pd.Series([1,3,5,8,10,11,15])
print("Original Series:\n",sr1)

first_diff = sr1.diff()
diff_of_diff = first_diff.diff()

print("\nFirst Difference:\n",first_diff)
print("\nSecond Difference:\n",diff_of_diff)

print("---------------------------------------------------")
print("\nQues27:\n")

sr1 = pd.Series(["01 Jan 2015","10-02-2016","20180307","2014/05/06","2016-04-12","2019-04-06T11:20"])
print("Original Series:\n",sr1)

sr2 = pd.to_datetime(sr1,format='mixed')
print("\nSeries of date strings to a timeseries:\n",sr2)

print("---------------------------------------------------")
print("\nQues28:\n")

sr1 = pd.Series(["01 Jan 2015","10-02-2016","20180307","2014/05/06","2016-04-12","2019-04-06T11:20"])

print("Original Series:\n",sr1)

sr2 = pd.to_datetime(sr1,format='mixed')

sr3 = sr2.dt.year
print("\nYear:\n",sr3)
sr4 = sr2.dt.month
print("\nDay of month:\n",sr4)
sr5 = sr2.dt.day
print("\nDates :\n",sr5)
sr6 = sr2.dt.isocalendar()['week']
print("\nWeek number::\n",sr6)
sr7 = sr2.dt.day_name()

print("Day of week:\n",sr7)


print("---------------------------------------------------")
print("\nQues29:\n")

sr1 = pd.Series(['Jan 2015','Feb 2016','Mar 2017','Apr 2018','May 2019'])

# Specify the day of the month you want to add
day = 11

# Convert to dates by adding the specified day
dates = pd.to_datetime([f"{ym}-{day}" for ym in sr1])

print("Original Series:\n",sr1)
print("\nNew dates:\n",dates)

print("---------------------------------------------------")
print("\nQues30:\n")

sr1 = pd.Series(['Red','Green','Orange','Pink','Yellow','White'])
print("Original Series:\n",sr1)


# Function to count vowels in a word
def has_two_vowels(sr1):
    vowels = 'aeiou'
    return sum(1 for char in sr1 if char in vowels) >= 2

# Filter the Series
filtered_words = sr1[sr1.apply(has_two_vowels)]

print("\nSeries after filtering:\n",filtered_words)

print("---------------------------------------------------")
print("\nQues31:\n")

sr1 = pd.Series([1,2,3,4,5,6,7,8,9,10])
sr2 = pd.Series([11,8,7,5,6,5,3,4,7,1])

print("First Series:\n",sr1)
print("Second Series:\n",sr2)

# Function to compute Euclidean distance
def euclidean_distance(s1, s2):
    return np.sqrt(((s1 - s2) ** 2).sum())

# Calculate the distance
distance = euclidean_distance(sr1, sr2)

# Display the result
print(f'Euclidean Distance: {distance}')


print("---------------------------------------------------")
print("\nQues32:\n")

sr1 = pd.Series([1,8,7,5,6,5,3,4,7,1])
print("Original Series:\n",sr1)

# Finding positions of values that are surrounded by smaller values
def find_local_minima(sr1):
    # Create a boolean mask for values that are greater than their neighbors
    mask = (sr1 > sr1.shift(1)) & (sr1 > sr1.shift(-1))

    #sr1.shift(1):This method shifts the Series down by one position.
    #sr1.shift(-1):This method shifts the Series up by one position.

    # Get the positions of local minima
    return sr1.index[mask].tolist()

# Find the positions
positions = find_local_minima(sr1)

# Display the result
print(f'Positions of values neighbored by smaller values: {positions}')


print("---------------------------------------------------")
print("\nQues33:\n")

sr1 = pd.Series(['abc def abcdef icd'])

print("Original Series:\n",sr1)
# Function to replace spaces with the least frequent character
def replace_spaces_with_least_frequent_char(series):
    # Convert Series to a single string
    input_string = series.iloc[0]

    # Count frequency of each character (excluding spaces)
    frequency = Counter(input_string.replace(" ", ""))

    # Find the least frequent character
    least_frequent_char = min(frequency, key=frequency.get)

    # Replace spaces with the least frequent character
    modified_string = input_string.replace(" ", least_frequent_char)

    return modified_string


# Replace spaces in the input Series string
result = replace_spaces_with_least_frequent_char(sr1)

# Display the result
print(f'Modified String: "{result}"')


print("---------------------------------------------------")
print("\nQues34:\n")

data = pd.Series(np.random.rand(14))

print("Original Series:\n",data)

# Function to compute autocorrelations for multiple lags
def compute_autocorrelations(series, max_lag):
    autocorrelations = {}
    for lag in range(1, max_lag + 1):
        autocorr = series.autocorr(lag=lag)
        autocorrelations[lag] = autocorr
    return autocorrelations

# Set the maximum lag
max_lag = 10

# Compute autocorrelations
autocorr_results = compute_autocorrelations(data, max_lag)

# Display the results
print("Autocorrelations:")
for lag, value in autocorr_results.items():
    print(f"Lag {lag}: {value}")


print("---------------------------------------------------")
print("\nQues35:\n")

# Specify the year for which you want to find all Sundays
year = 2023

# Create a date range for the entire year, with frequency set to 'W-SUN' for Sundays
sundays = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='W-SUN')

# Convert the date range to a pandas Series
sundays_series = pd.Series(sundays)

# Display the result
print(f'All Sundays in {year}:')
print(sundays_series)



print("---------------------------------------------------")
print("\nQues36:\n")

# Sample data: create a pandas Series
data = pd.Series(['A', 'B', 'C', 'D', 'E'])

# Convert Series to DataFrame and reset the index
df = data.reset_index()

# Rename the columns for clarity
df.columns = ['Index', 'Value']

# Display the result
print(df)

print(df.to_string(index=False))

print("---------------------------------------------------")
print("\nQues37:\n")


# Create two sample Series
series1 = pd.Series(['A', 'B', 'C'])
series2 = pd.Series(['D', 'E', 'F'])

# Stack Vertically
vertical_stack = pd.concat([series1, series2], axis=0).reset_index(drop=True)

# Stack Horizontally
horizontal_stack = pd.concat([series1, series2], axis=1)

# Display the results
print("Vertical Stack:")
print(vertical_stack)

print("\nHorizontal Stack:")
print(horizontal_stack)

print("---------------------------------------------------")
print("\nQues38:\n")

# Create two sample Series
series1 = pd.Series([1, 2, 3, 4])
series2 = pd.Series([1, 2, 3, 4])
series3 = pd.Series([1, 2, 3, 5])

# Check equality of series1 and series2
are_equal1 = series1.equals(series2)

# Check equality of series1 and series3
are_equal2 = series1.equals(series3)

# Display the results
print(f'Are series1 and series2 equal? {are_equal1}')
print(f'Are series1 and series3 equal? {are_equal2}')

print("---------------------------------------------------")
print("\nQues39:\n")


# Create a sample Series
data = pd.Series([5, 3, 1, 4, 2, 1, 6])

# Find the index of the first occurrence of the smallest value
index_of_smallest = data.idxmin()

# Find the index of the first occurrence of the largest value
index_of_largest = data.idxmax()

# Display the results
print(f'Index of the first occurrence of the smallest value: {index_of_smallest}')
print(f'Index of the first occurrence of the largest value: {index_of_largest}')

print("---------------------------------------------------")
print("\nQues40:\n")

## Create the original Series
original_series = pd.Series([68.0, 75.0, 86.0, 80.0, np.nan])

# Create a sample DataFrame with values for comparison
data = {
    'W': [68.0, 75.0, 86.0, 80.0, np.nan],
    'X': [70.0, 75.0, 90.0, 85.0, 100.0],
    'Y': [68.0, 80.0, 86.0, 78.0, 90.0],
    'Z': [70.0, 75.0, 87.0, 79.0, 95.0]
}
df = pd.DataFrame(data)

# Check for inequality over the index axis
inequality_result = df.ne(original_series, axis=0)

# Display the results
print("Original Series:")
print(original_series)

print("\nCheck for inequality of the said series & dataframe:")
print(inequality_result)