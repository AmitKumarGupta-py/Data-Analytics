import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


print("---------------------------------------------------------------------------------")
print("Code 1")

data = {
    'ord_no': [70001.0, np.nan, 70002.0, 70004.0, np.nan, 70005.0,
               np.nan, 70010.0, 70003.0, 70012.0, np.nan, 70013.0],
    'purch_amt': [150.50, 270.65, 65.26, 110.50, 948.50, 2400.60,
                  5760.00, 1983.43, 2480.40, 250.45, 75.29, 3045.60],
    'ord_date': ['2012-10-05', '2012-09-10', np.nan, '2012-08-17',
                 '2012-09-10', '2012-07-27', '2012-09-10',
                 '2012-10-10', '2012-10-10', '2012-06-27',
                 '2012-08-17', '2012-04-25'],
    'customer_id': [3002, 3001, 3001, 3003, 3002, 3001, 3001,
                    3004, 3003, 3002, 3001, 3001],
    'salesman_id': [5002.0, 5003.0, 5001.0, np.nan, 5002.0,
                    5001.0, 5001.0, np.nan, 5003.0, 5002.0,
                    5003.0, np.nan]
}

df = pd.DataFrame(data)

print("\nOriginal Data:\n",df)

df1 = df.isna()
print("\nto detect missing values of a given DataFrame and Display True or False:\n\n",df1)


print("---------------------------------------------------------------------------------")
print("Code 2")

df1 = df.copy(deep=True)

print("\nOriginal Data:\n",df1,"\n")
missing_columns = df.columns[df.isna().any()]
print("\nThe column(s) of a given DataFrame which have at least one missing value:\n\n",missing_columns)





print("---------------------------------------------------------------------------------")
print("Code 3")

missing_count = df1.isna().sum()

print("\nTo count the number of missing values in each column of a given DataFrame:\n\n",missing_count)





print("---------------------------------------------------------------------------------")
print("Code 4")


data = {
    'ord_no': [70001, np.nan, 70002, 70004, np.nan, 70005,
               '--', 70010, 70003, 70012, np.nan, 70013],
    'purch_amt': [150.5, 270.65, 65.26, 110.5, 948.5, 2400.6,
                  '?', 5760, 12.43, 2480.4, 250.45, 3045.6],
    'ord_date': ['?', '2012-09-10', np.nan, '2012-08-17',
                 '2012-09-10', '2012-07-27', '2012-09-10',
                 '2012-10-10', '2012-10-10', '2012-06-27',
                 '2012-08-17', '2012-04-25'],
    'customer_id': [3002, 3001, 3001, 3003, 3002, 3001,
                    3004, 3004, '--', 3002, 3001, 3001],
    'salesman_id': [5002, 5003, '?', 5001, np.nan, 5002,
                    '?', '?', 5003, 5002, '5003', '--']
}

df = pd.DataFrame(data)

print("\nOriginal Data:\n\n",df)

df.replace({'?': np.nan, '--': np.nan}, inplace=True)

print("Modified Data:\n",df)

# Convert relevant columns to numeric where appropriate
df['ord_no'] = pd.to_numeric(df['ord_no'], errors='coerce')
df['ord_date'] =pd.to_datetime(df['ord_date'],errors = 'coerce' )
df['purch_amt'] = pd.to_numeric(df['purch_amt'], errors='coerce')
df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce')
df['salesman_id'] = pd.to_numeric(df['salesman_id'], errors='coerce')


# Fill with the mean for numeric columns
df['ord_no'].fillna(df['ord_no'].mean(), inplace=True)
df['purch_amt'].fillna(df['purch_amt'].mean(), inplace=True)
df['ord_date'] = df['ord_date'].bfill()
df['customer_id'].fillna(df['customer_id'].mode()[0], inplace=True)  # Mode for categorical
df['salesman_id'].fillna(df['salesman_id'].mode()[0], inplace=True)  # Mode for categorical


print("DataFrame after filling missing values:")
print(df)


print("---------------------------------------------------------------------------------")
print("Code 5")


data = {
    'ord_no': [70001.0, np.nan, 70002.0, 70004.0, np.nan, 70005.0,
               np.nan, 70010.0, 70003.0, 70012.0, np.nan, 70013.0],
    'purch_amt': [150.50, 270.65, 65.26, 110.50, 948.50, 2400.60,
                  5760.00, 1983.43, 2480.40, 250.45, 75.29, 3045.60],
    'ord_date': ['2012-10-05', '2012-09-10', np.nan, '2012-08-17',
                 '2012-09-10', '2012-07-27', '2012-09-10',
                 '2012-10-10', '2012-10-10', '2012-06-27',
                 '2012-08-17', '2012-04-25'],
    'customer_id': [3002, 3001, 3001, 3003, 3002, 3001, 3001,
                    3004, 3003, 3002, 3001, 3001],
    'salesman_id': [5002.0, 5003.0, 5001.0, np.nan, 5002.0,
                    5001.0, 5001.0, np.nan, 5003.0, 5002.0,
                    5003.0, np.nan]
}

df = pd.DataFrame(data)
print("\nOriginal Data:\n\n",df)

df1 = df.dropna()

print("\n\nCleaned Data:\n\n",df1)


print("---------------------------------------------------------------------------------")
print("Code 6")

data = {
    'ord_no': [70001.0, np.nan, 70002.0, 70004.0, np.nan, 70005.0,
               np.nan, 70010.0, 70003.0, 70012.0, np.nan, 70013.0],
    'purch_amt': [150.50, 270.65, 65.26, 110.50, 948.50, 2400.60,
                  5760.00, 1983.43, 2480.40, 250.45, 75.29, 3045.60],
    'ord_date': ['2012-10-05', '2012-09-10', np.nan, '2012-08-17',
                 '2012-09-10', '2012-07-27', '2012-09-10',
                 '2012-10-10', '2012-10-10', '2012-06-27',
                 '2012-08-17', '2012-04-25'],
    'customer_id': [3002, 3001, 3001, 3003, 3002, 3001, 3001,
                    3004, 3003, 3002, 3001, 3001],
    'salesman_id': [5002.0, 5003.0, 5001.0, np.nan, 5002.0,
                    5001.0, 5001.0, np.nan, 5003.0, 5002.0,
                    5003.0, np.nan]
}

df = pd.DataFrame(data)
print("\nOriginal Data:\n\n",df)

df1 = df.dropna(axis=1)

print("\n\nCleaned Data:\n\n",df1)


print("---------------------------------------------------------------------------------")
print("Code 7")

data = {
    'ord_no': [np.nan, np.nan, 70002.0, 70004.0, np.nan, 70005.0,
               np.nan, 70010.0, 70003.0, 70012.0, np.nan, 70013.0],
    'purch_amt': [np.nan, 270.65, 65.26, 110.50, 948.50, 2400.60,
                  5760.00, 1983.43, 2480.40, 250.45, 75.29, 3045.60],
    'ord_date': [np.nan, '2012-09-10', np.nan, '2012-08-17',
                 '2012-09-10', '2012-07-27', '2012-09-10',
                 '2012-10-10', '2012-10-10', '2012-06-27',
                 '2012-08-17', '2012-04-25'],
    'customer_id': [np.nan, 3001.0, 3001.0, 3003.0, 3002.0, 3001.0,
                    3001.0, 3004.0, 3003.0, 3002.0, 3001.0, 3001.0]
}

df = pd.DataFrame(data)

print("\n\nOriginal Data:\n\n",df)

df1 = df.dropna(how = 'all')

print("\n\nCleaned Data:\n\n",df1)


print("---------------------------------------------------------------------------------")
print("Code 8")

data = {
    'ord_no': [np.nan, np.nan, 70002.0, np.nan, np.nan, 70005.0, np.nan, 70010.0, 70003.0, 70012.0, np.nan, np.nan],
    'purch_amt': [np.nan, 270.65, 65.26, np.nan, 948.50, 2400.60, 5760.00, 1983.43, 2480.40, 250.45, 75.29, np.nan],
    'ord_date': [np.nan, '2012-09-10', np.nan, np.nan, '2012-09-10', '2012-07-27', '2012-09-10', '2012-10-10', '2012-10-10', '2012-06-27', '2012-08-17', np.nan],
    'customer_id': [np.nan, 3001.0, 3001.0, np.nan, 3002.0, 3001.0, 3001.0, 3004.0, 3003.0, 3002.0, 3001.0, np.nan]
}

df = pd.DataFrame(data)

df1 =df.copy(deep = True)

print("\n\nOriginal Data:\n\n",df1)
df1 = df1[(df1.isna().sum(axis = 1)) >= 2]

print("\n\nCleaned Data:\n\n",df1)

print("---------------------------------------------------------------------------------")
print("Code 9")

data = {
    'ord_no': [np.nan, np.nan, 70002.0, np.nan, np.nan, 70005.0, np.nan, 70010.0, 70003.0, 70012.0, np.nan, np.nan],
    'purch_amt': [np.nan, 270.65, 65.26, np.nan, 948.50, 2400.60, 5760.00, 1983.43, 2480.40, 250.45, 75.29, np.nan],
    'ord_date': [np.nan, '2012-09-10', np.nan, np.nan, '2012-09-10', '2012-07-27', '2012-09-10', '2012-10-10', '2012-10-10', '2012-06-27', '2012-08-17', np.nan],
    'customer_id': [np.nan, 3001.0, 3001.0, np.nan, 3002.0, 3001.0, 3001.0, 3004.0, 3003.0, 3002.0, 3001.0, np.nan]
}

df = pd.DataFrame(data)

df1 =df.copy(deep = True)

print("\n\nOriginal Data:\n\n",df1)
df1 = df1.dropna()

print("\n\nCleaned Data:\n\n",df1)

print("---------------------------------------------------------------------------------")
print("Code 10")

data = {
    'ord_no': [np.nan, np.nan, 70002.0, np.nan, np.nan, 70005.0, np.nan, 70010.0, 70003.0, 70012.0, np.nan, np.nan],
    'purch_amt': [np.nan, 270.65, 65.26, np.nan, 948.50, 2400.60, 5760.00, 1983.43, 2480.40, 250.45, 75.29, np.nan],
    'ord_date': [np.nan, '2012-09-10', np.nan, np.nan, '2012-09-10', '2012-07-27', '2012-09-10', '2012-10-10', '2012-10-10', '2012-06-27', '2012-08-17', np.nan],
    'customer_id': [np.nan, 3001.0, 3001.0, np.nan, 3002.0, 3001.0, 3001.0, 3004.0, 3003.0, 3002.0, 3001.0, np.nan]
}

df = pd.DataFrame(data)

df1 =df.copy(deep = True)

print("\n\nOriginal Data:\n\n",df1)
df1 = df1.dropna()

print("\n\nCleaned Data:\n\n",df1)


print("---------------------------------------------------------------------------------")
print("Code 11")

data = {
    'ord_no': [np.nan, np.nan, 70002.0, np.nan, np.nan, 70005.0, np.nan, 70010.0, 70003.0, 70012.0, np.nan, np.nan],
    'purch_amt': [np.nan, 270.65, 65.26, np.nan, 948.50, 2400.60, 5760.00, 1983.43, 2480.40, 250.45, 75.29, np.nan],
    'ord_date': [np.nan, '2012-09-10', np.nan, np.nan, '2012-09-10', '2012-07-27', '2012-09-10', '2012-10-10', '2012-10-10', '2012-06-27', '2012-08-17', np.nan],
    'customer_id': [np.nan, 3001.0, 3001.0, np.nan, 3002.0, 3001.0, 3001.0, 3004.0, 3003.0, 3002.0, 3001.0, np.nan]
}

df = pd.DataFrame(data)

df1 =df.copy(deep = True)

print("\n\nOriginal Data:\n\n",df1)
df1 = df1.isna().sum()

print("\n\nthe total number of missing values in each column of a Data:\n\n",df1)

print("\n\n total no. of missing values in Data:\n\n",df1.sum())


print("---------------------------------------------------------------------------------")
print("Code 12")

data = {
    'ord_no': [np.nan, np.nan, 70002.0, np.nan, np.nan, 70005.0, np.nan, 70010.0, 70003.0, 70012.0, np.nan, np.nan],
    'purch_amt': [np.nan, 270.65, 65.26, np.nan, 948.50, 2400.60, 5760.00, 1983.43, 2480.40, 250.45, 75.29, np.nan],
    'ord_date': [np.nan, '2012-09-10', np.nan, np.nan, '2012-09-10', '2012-07-27', '2012-09-10', '2012-10-10', '2012-10-10', '2012-06-27', '2012-08-17', np.nan],
    'customer_id': [np.nan, 3001.0, 3001.0, np.nan, 3002.0, 3001.0, 3001.0, 3004.0, 3003.0, 3002.0, 3001.0, np.nan]
}

df = pd.DataFrame(data)
print("\n\nOriginal Data:\n\n",df)

df = df.fillna(0)
print("\n\nto replace NaNs with a single constant value in specified columns in a DataFrame:\n\n",df)

print("---------------------------------------------------------------------------------")
print("Code 13")

data = {
    'ord_no': [70001.0, np.nan, 70002.0, 70004.0, np.nan, 70005.0, np.nan, 70010.0, 70003.0, 70012.0, np.nan, 70013.0],
    'purch_amt': [150.50, np.nan, 65.26, 110.50, 948.50, np.nan, 5760.00, 1983.43, np.nan, 250.45, 75.29, 3045.60],
    'sale_amt': [10.50, 20.65, np.nan, 11.50, 98.50, np.nan, 57.00, 19.43, np.nan, 25.45, 75.29, 35.60],
    'ord_date': ['2012-10-05', '2012-09-10', np.nan, '2012-08-17', '2012-09-10', '2012-07-27', '2012-09-10', '2012-10-10', '2012-10-10', '2012-06-27', '2012-08-17', '2012-04-25'],
    'customer_id': [3002, 3001, 3001, 3003, 3002, 3001, 3001, 3004, 3003, 3002, 3001, 3001],
}

df = pd.DataFrame(data)

df1 = df.copy(deep=True)
df2 = df.copy(deep = True)
print("\n\nOriginal Data:\n\n",df1)

print("\n\nCleaned Data with froward filling:\n\n",df1.ffill())

print("\n\nCleaned Data with Backward filling:\n\n",df2.bfill())

print("---------------------------------------------------------------------------------")
print("Code 14")


data = {
    'ord_no': [70001.0, np.nan, 70002.0, 70004.0, np.nan, 70005.0, np.nan, 70010.0, 70003.0, 70012.0, np.nan, 70013.0],
    'purch_amt': [150.50, np.nan, 65.26, 110.50, 948.50, np.nan, 5760.00, 1983.43, np.nan, 250.45, 75.29, 3045.60],
    'sale_amt': [10.50, 20.65, np.nan, 11.50, 98.50, np.nan, 57.00, 19.43, np.nan, 25.45, 75.29, 35.60],
    'ord_date': ['2012-10-05', '2012-09-10', np.nan, '2012-08-17', '2012-09-10', '2012-07-27', '2012-09-10', '2012-10-10', '2012-10-10', '2012-06-27', '2012-08-17', '2012-04-25'],
    'customer_id': [3002, 3001, 3001, 3003, 3002, 3001, 3001, 3004, 3003, 3002, 3001, 3001],
}

df = pd.DataFrame(data)

df1 = df.copy(deep = True)
df2 = df.copy(deep = True)

print("\n\nOriginal Data:\n\n",df1)

df1.fillna(df1.mean(numeric_only=True), inplace=True)


print("\n\nCleaned Data with mean Filling:\n\n",df1)

df2.fillna(df1.median(numeric_only=True),inplace = True)

print("\n\n Cleaned data with median filling\n\n:",df2)

print("---------------------------------------------------------------------------------")
print("Code 15")
data = {
    'ord_no': [70001.0, np.nan, 70002.0, 70004.0, np.nan, 70005.0, np.nan, 70010.0, 70003.0, 70012.0, np.nan, 70013.0],
    'purch_amt': [150.50, np.nan, 65.26, 110.50, 948.50, np.nan, 5760.00, 1983.43, np.nan, 250.45, 75.29, 3045.60],
    'sale_amt': [10.50, 20.65, np.nan, 11.50, 98.50, np.nan, 57.00, 19.43, np.nan, 25.45, 75.29, 35.60],
    'ord_date': ['2012-10-05', '2012-09-10', np.nan, '2012-08-17', '2012-09-10', '2012-07-27', '2012-09-10', '2012-10-10', '2012-10-10', '2012-06-27', '2012-08-17', '2012-04-25'],
    'customer_id': [3002, 3001, 3001, 3003, 3002, 3001, 3001, 3004, 3003, 3002, 3001, 3001],
}

df = pd.DataFrame(data)

df1 = df.copy(deep=True)
print("\n\nOriginal data:\n\n",df1)

df1 = df1.interpolate(method='linear',limit_direction='forward')

print("\n\nCleaned Data using interpolation:\n\n",df1)


print("---------------------------------------------------------------------------------")
print("Code 16")

data = {
    'ord_no': [70001.0, np.nan, 70002.0, 70004.0, np.nan, 70005.0, np.nan, 70010.0, 70003.0, 70012.0, np.nan, 70013.0],
    'purch_amt': [150.50, np.nan, 65.26, 110.50, 948.50, np.nan, 5760.00, 1983.43, np.nan, 250.45, 75.29, 3045.60],
    'sale_amt': [10.50, 20.65, np.nan, 11.50, 98.50, np.nan, 57.00, 19.43, np.nan, 25.45, 75.29, 35.60],
    'ord_date': ['2012-10-05', '2012-09-10', np.nan, '2012-08-17', '2012-09-10', '2012-07-27', '2012-09-10', '2012-10-10', '2012-10-10', '2012-06-27', '2012-08-17', '2012-04-25'],
    'customer_id': [3002, 3001, 3001, 3003, 3002, 3001, 3001, 3004, 3003, 3002, 3001, 3001],
}

df = pd.DataFrame(data)

df1 = df.copy(deep=True)
print("\n\nOriginal data:\n\n",df1)

df1 = df1.isna().sum()
print("\n\nthe number of missing values of a specified column in a given DataFrame.:\n\n",df1)

print("---------------------------------------------------------------------------------")
print("Code 17")

data = {
    'ord_no': [70001.0, np.nan, 70002.0, 70004.0, np.nan, 70005.0, np.nan, 70010.0, 70003.0, 70012.0, np.nan, 70013.0],
    'purch_amt': [150.50, np.nan, 65.26, 110.50, 948.50, np.nan, 5760.00, 1983.43, np.nan, 250.45, 75.29, 3045.60],
    'sale_amt': [10.50, 20.65, np.nan, 11.50, 98.50, np.nan, 57.00, 19.43, np.nan, 25.45, 75.29, 35.60],
    'ord_date': ['2012-10-05', '2012-09-10', np.nan, '2012-08-17', '2012-09-10', '2012-07-27', '2012-09-10', '2012-10-10', '2012-10-10', '2012-06-27', '2012-08-17', '2012-04-25'],
    'customer_id': [3002, 3001, 3001, 3003, 3002, 3001, 3001, 3004, 3003, 3002, 3001, 3001],
}

df = pd.DataFrame(data)

df1 = df.copy(deep=True)
print("\n\nOriginal data:\n\n",df1)

df1 = df1.isna().sum()
print("\n\nthe number of missing values of a specified column in a given DataFrame.:\n\n",df1)

count_missing = df1.sum()
print("\n\n Total no. of missing values:\n\n",count_missing)

print("---------------------------------------------------------------------------------")
print("Code 18")

data = {
    'ord_no': [70001.0, np.nan, 70002.0, 70004.0, np.nan, 70005.0, np.nan, 70010.0, 70003.0, 70012.0, np.nan, 70013.0],
    'purch_amt': [150.50, np.nan, 65.26, 110.50, 948.50, np.nan, 5760.00, 1983.43, np.nan, 250.45, 75.29, 3045.60],
    'sale_amt': [10.50, 20.65, np.nan, 11.50, 98.50, np.nan, 57.00, 19.43, np.nan, 25.45, 75.29, 35.60],
    'ord_date': ['2012-10-05', '2012-09-10', np.nan, '2012-08-17', '2012-09-10', '2012-07-27', '2012-09-10', '2012-10-10', '2012-10-10', '2012-06-27', '2012-08-17', '2012-04-25'],
    'customer_id': [3002, 3001, 3001, 3003, 3002, 3001, 3001, 3004, 3003, 3002, 3001, 3001],
}

df = pd.DataFrame(data)

df1 = df.copy(deep=True)
print("\n\nOriginal data:\n\n",df1)

missing_value_index = df[df.isna().any(axis=1)].index


print("\n\nthe Indexes of missing values in a given DataFrame.:\n\n",missing_value_index.tolist())

print("---------------------------------------------------------------------------------")
print("Code 19")

data = {
    'ord_no': [70001.0, np.nan, 70002.0, 70004.0, np.nan, 70005.0, np.nan, 70010.0, 70003.0, 70012.0, np.nan, 70013.0],
    'purch_amt': [150.50, np.nan, 65.26, 110.50, 948.50, np.nan, 5760.00, 1983.43, np.nan, 250.45, 75.29, 3045.60],
    'sale_amt': [10.50, 20.65, np.nan, 11.50, 98.50, np.nan, 57.00, 19.43, np.nan, 25.45, 75.29, 35.60],
    'ord_date': ['2012-10-05', '2012-09-10', np.nan, '2012-08-17', '2012-09-10', '2012-07-27', '2012-09-10', '2012-10-10', '2012-10-10', '2012-06-27', '2012-08-17', '2012-04-25'],
    'customer_id': [3002, 3001, 3001, 3003, 3002, 3001, 3001, 3004, 3003, 3002, 3001, 3001],
}

df = pd.DataFrame(data)

df1 = df.copy(deep=True)
print("\n\nOriginal data:\n\n",df1)

for column in df1.columns:
    mode_value = df1[column].mode()[0]
    df1[column].fillna(mode_value, inplace=True)


print("\n\nreplace the missing values with the most frequent values present in each column of a given DataFrame:\n\n",df1)

print("---------------------------------------------------------------------------------")
print("Code 20")

data = {
    'ord_no': [70001.0, np.nan, 70002.0, 70004.0, np.nan, 70005.0, np.nan, 70010.0, 70003.0, 70012.0, np.nan, 70013.0],
    'purch_amt': [150.50, np.nan, 65.26, 110.50, 948.50, np.nan, 5760.00, 1983.43, np.nan, 250.45, 75.29, 3045.60],
    'sale_amt': [10.50, 20.65, np.nan, 11.50, 98.50, np.nan, 57.00, 19.43, np.nan, 25.45, 75.29, 35.60],
    'ord_date': ['2012-10-05', '2012-09-10', np.nan, '2012-08-17', '2012-09-10', '2012-07-27', '2012-09-10', '2012-10-10', '2012-10-10', '2012-06-27', '2012-08-17', '2012-04-25'],
    'customer_id': [3002, 3001, 3001, 3003, 3002, 3001, 3001, 3004, 3003, 3002, 3001, 3001],
}

df = pd.DataFrame(data)

df1 = df.copy(deep=True)

print("\n\nOriginal data:\n\n",df1)


df2 = df1.isna()
print(df2)
plt.figure(figsize=(8, 6))
sns.heatmap(df1.isna(), cmap='viridis', cbar=False, yticklabels=False, linewidths=0.5)

plt.title('Missing Values Heatmap')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.tight_layout()
plt.show()