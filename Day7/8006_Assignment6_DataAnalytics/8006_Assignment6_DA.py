import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
from stack_data import markers_from_ranges

print("---------------------------------------------------------------------------")
print("Ques 1")
df = pd.read_csv("alphabet_stock_data.csv")
print(df.info())

print(df.head())
date_series = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
df1 = df.head(10)
df1['Date'] = date_series


print(df1.head(10))
plt.figure(figsize= (8,6))
sns.lineplot(x = 'Date',y = 'High',data = df1)
plt.xlabel('Date')
plt.ylabel('Stocks')
plt.title('Alphabet Stock ')
plt.tight_layout()
plt.xticks(rotation = 45)
plt.show()

print("---------------------------------------------------------------------------")
print("Ques 2")

plt.figure(figsize= (8,6))
sns.lineplot(x = 'Date',y = 'Open',label = 'Open',data = df1)
sns.lineplot(x = 'Date', y = 'Close',label = 'Close', data = df1)
plt.xlabel('Date')
plt.ylabel('Stocks')
plt.title('Alphabet Stock ')
plt.legend()
plt.tight_layout()
plt.xticks(rotation = 45)
plt.show()


print("---------------------------------------------------------------------------")
print("Ques 3")

plt.figure(figsize= (8,6))
sns.barplot(x = 'Date',y = 'Volume',data = df1, palette= 'viridis')

plt.xlabel('Date')
plt.ylabel('Volume of Stocks')
plt.title('Alphabet Stock ')
plt.tight_layout()
plt.xticks(rotation = 45)
plt.show()

print("---------------------------------------------------------------------------")
print("Ques 4")

melted_data = pd.melt(df1, id_vars='Date', value_vars=['Open', 'Close'],
                       var_name='Price Type', value_name='Price')

plt.figure(figsize= (8,6))
sns.barplot(x = 'Date',y = 'Price',hue = 'Price Type', data = melted_data, palette = 'deep')


plt.xlabel('Date')
plt.ylabel('Volume of Stocks')
plt.title('Alphabet Stock ')
plt.legend(title = 'Price Type')
plt.tight_layout()
plt.xticks(rotation = 45)
plt.show()


print("---------------------------------------------------------------------------")
print("Ques 5")

#melted_data = pd.melt(df1, id_vars='Date', value_vars=['Open', 'Close'], var_name='Price Type', value_name='Price')

data_subset = df1[['Date','Open','Close']]
data_subset.set_index('Date',inplace = True)

plt.figure(figsize= (8,6))
data_subset.plot(kind='bar', stacked=True, color=['skyblue', 'lightgreen'], edgecolor='black')


plt.xlabel('Date')
plt.ylabel('Opening and Closing Stock Prices of Alphabet Inc.')
plt.title('Alphabet Stock ')
plt.legend(title = 'Price Type',labels=['Open','Close'])
plt.tight_layout()
plt.xticks(rotation = 45)
plt.show()


print("---------------------------------------------------------------------------")
print("Ques 6")

data_subset = df1[['Date','Open','Close']]
data_subset.set_index('Date',inplace = True)

plt.figure(figsize= (8,6))
data_subset.plot(kind='barh', stacked=True, color=['skyblue', 'lightgreen'], edgecolor='black')


plt.xlabel('Opening and Closing Stock Prices of Alphabet Inc.')
plt.ylabel('Date')
plt.title('Alphabet Stock ')
plt.legend(title = 'Price Type',labels=['Open','Close'])
plt.tight_layout()
plt.xticks(rotation = 45)
plt.show()


print("---------------------------------------------------------------------------")
print("Ques 7")

data_subset = df1[['Date','Open','Close','High']]
data_subset.set_index('Date',inplace = True)

plt.figure(figsize= (8,6))
plt.hist(data_subset , bins = 10, color = ['skyblue','blue','red'], edgecolor = 'black', alpha = 0.7)


plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Alphabet Stock ')
plt.grid(axis='y',alpha = 0.75)
plt.tight_layout()
plt.xticks(rotation = 45)
plt.show()


print("---------------------------------------------------------------------------")
print("Ques 8")

data_subset = df1[['Date','Open','Close','High']]
data_subset.set_index('Date',inplace = True)

plt.figure(figsize= (8,6))
plt.hist(data_subset ,stacked = True, bins = 10, color = ['skyblue','blue','red'], edgecolor = 'black', alpha = 0.7)


plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Alphabet Stock ')
plt.grid(axis='y',alpha = 0.75)
plt.tight_layout()
plt.xticks(rotation = 45)
plt.show()

print("---------------------------------------------------------------------------")
print("Ques 9")

data_subset = df1[['Date','Open','Close','High']]
data_subset.set_index('Date',inplace = True)

plt.figure(figsize= (8,6))
plt.hist(data_subset ,cumulative = True, bins = 10, color = ['skyblue','blue','red'], edgecolor = 'black', alpha = 0.7)


plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Cumulative Histogram of Alphabet Stock ')
plt.grid(axis='y',alpha = 0.75)
plt.legend(title = 'Price Type',labels=['Open','Close','High'])
plt.tight_layout()
plt.xticks(rotation = 45)
plt.show()


print("---------------------------------------------------------------------------")
print("Ques 10")

data_subset = df1[['Date','Open','Close','High','Low']]

data_subset.set_index('Date',inplace = True)

plt.figure(figsize= (8,6))
plt.hist(data_subset ,stacked = True, color = ['skyblue','blue','red','green'], edgecolor = 'black', alpha = 0.7)


plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Stacked Histogram of Alphabet Stock ')
plt.grid(axis='y',alpha = 0.75)
plt.tight_layout()
plt.legend(title = 'Price Type',labels=['Open','Close','High','Low'])
plt.xticks(rotation = 45)
plt.show()

print("---------------------------------------------------------------------------")
print("Ques 11")

data_subset = df1[['Date','Open','Close','High','Low']]

data_subset.set_index('Date',inplace = True)

plt.figure(figsize= (8,6))
plt.hist(data_subset ,stacked = True, bins = 20, color = ['skyblue','blue','red','green'], edgecolor = 'black', alpha = 0.7)


plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Stacked Histogram of Alphabet Stock ')
plt.grid(axis='y',alpha = 0.75)
plt.tight_layout()
plt.legend(title = 'Price Type',labels=['Open','Close','High','Low'])
plt.xticks(rotation = 45)
plt.show()


print("---------------------------------------------------------------------------")
print("Ques 12")

print(df1.columns)


#data = df(stock_symbol, start=start_date, end=end_date)

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(df1.index, df1['Close'], color='blue', label='Stock Price', linewidth=2)
ax1.set_xlabel('Date')
ax1.set_ylabel('Stock Price (USD)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.bar(df1.index, df1['Volume'], color='purple', alpha=0.3, label='Trading Volume')
ax2.set_ylabel('Trading Volume', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

plt.title(f'Stock Price and Trading Volume of Alphabet')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()

plt.show()


print("---------------------------------------------------------------------------")
print("Ques 13")

fig, ax1 = plt.subplots( figsize=(8, 6))


ax1.plot(df1.index, df1['Open'], label='Open', color='blue', linewidth=1)
ax1.plot(df1.index, df1['High'], label='High', color='green', linewidth=1)
ax1.plot(df1.index, df1['Low'], label='Low', color='red', linewidth=1)
ax1.plot(df1.index, df1['Close'], label='Close', color='orange', linewidth=1)
ax1.plot(df1.index, df1['Adj Close'], label='Adjusted Close',marker ='x', color='purple', linewidth=1)
ax1.set_title(f'Alphabet Stock Prices')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price (USD)')
ax1.legend()

ax2 = ax1.twinx()

ax2.bar(df1.index, df1['Volume'], color='lightblue', alpha=0.3)

ax2.set_ylabel('Volume')


ax2.set_xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


print("---------------------------------------------------------------------------")
print("Ques 14")


df2 = df.head(100)


df2['30-Day SMA'] = df2['Adj Close'].rolling(window=30).mean()
df2['40-Day SMA'] = df2['Adj Close'].rolling(window=40).mean()

plt.figure(figsize=(8, 7))
plt.plot(df2.index, df2['Adj Close'], label='Adjusted Close', color='blue',marker = 's', linewidth=1.5)
plt.plot(df2.index, df2['30-Day SMA'], label='30-Day SMA', color='orange',linestyle ='--', linewidth=1.5)
plt.plot(df2.index, df2['40-Day SMA'], label='40-Day SMA', color='green',linestyle ='-', linewidth=1.5)

# Set title and labels
plt.title(f'Alphabet Adjusted Closing Prices with SMAs')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.xticks(rotation=45)
plt.grid()

plt.tight_layout()
plt.show()


print("---------------------------------------------------------------------------")
print("Ques 15")


df1 = df.head(100)

df1['30-Day SMA'] = df1['Adj Close'].rolling(window=30).mean()

df1['30-Day EMA'] = df1['Adj Close'].ewm(span=30, adjust=False).mean()


plt.figure(figsize=(10, 6))
plt.plot(df1.index, df1['Adj Close'], label='Adjusted Close', color='blue', linewidth=1.5)
plt.plot(df1.index, df1['30-Day SMA'], label='30-Day SMA', color='orange', linewidth=1.5)
plt.plot(df1.index, df1['30-Day EMA'], label='30-Day EMA', color='green', linewidth=1.5)


plt.title(f'Alphabet Adjusted Closing Prices with SMA and EMA')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.xticks(rotation=45)
plt.grid()

plt.tight_layout()
plt.show()


print("---------------------------------------------------------------------------")
print("Ques 16")


plt.figure(figsize=(8, 6))
plt.scatter(df1['Adj Close'], df1['Volume'], color='magenta', alpha=0.5)


plt.title(f'Scatter Plot of Trading Volume vs. Adjusted Closing Prices of Alphabet')
plt.xlabel('Adjusted Closing Prices (USD)')
plt.ylabel('Trading Volume')
plt.grid()


plt.tight_layout()
plt.show()

print("---------------------------------------------------------------------------")
print("Ques 17")

df1 = df.head(30)
df1['Daily Return'] = df1['Adj Close'].pct_change() * 100

plt.figure(figsize=(8, 7))
plt.plot(df1.index, df1['Daily Return'],marker = 's', color='blue', linewidth=1.5)

plt.title(f'Daily Percentage Returns of Alphabet')
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid()


plt.tight_layout()
plt.show()

print("---------------------------------------------------------------------------")
print("Ques 18")


df1 = df.head(120)

df1['Daily Return'] = df1['Adj Close'].pct_change()


window_size = 21
df1['Volatility'] = df1['Daily Return'].rolling(window=window_size).std() * 100


plt.figure(figsize=(8, 6))
plt.plot(df1.index, df1['Volatility'], color='blue', linewidth=1.5)


plt.title(f'Volatility of Alphabet Stock Price')
plt.xlabel('Date')
plt.ylabel('Volatility (%)')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid()

plt.tight_layout()
plt.show()

print("---------------------------------------------------------------------------")
print("Ques 19")

df1['Daily Return'] = df1['Adj Close'].pct_change() * 100


plt.figure(figsize=(8, 6))
plt.hist(df1['Daily Return'].dropna(), bins=30, color='olive', alpha=0.7)

plt.title(f'Daily Return Distribution of Alphabet')
plt.xlabel('Daily Return (%)')
plt.ylabel('Frequency')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')

plt.grid()
plt.tight_layout()
plt.show()

