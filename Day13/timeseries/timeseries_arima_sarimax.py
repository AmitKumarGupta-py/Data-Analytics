import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df =pd.read_csv("monthly-cola-production-in-austr.csv")
print(df.info())
print(df.columns)

#we get only year and month for the date , we need day also
#convert the date to a YYYY -MM-DD format in a new column named yearmonth

# df['yearMonth'] = pd.to_datetime("01-" + df['Month'].astype(str) )
# df.set_index('yearMonth',inplace = True)
print(df.info())
print(df.head())

plt.figure(figsize=(10,5))
sns.lineplot(data = df, x = df.index, y = df['Monthly cola production'])

plt.show()


df['rollMean'] = df['Monthly cola production'].rolling(window = 12).mean()
df['rollStd'] = df['Monthly cola production'].rolling(window = 12).std()

plt.figure(figsize=(10,5))
sns.lineplot(data = df, x = df.index, y = df.Month)
sns.lineplot(data=df,x= df.index, y = df.rollMean)
sns.lineplot(data = df, x= df.index, y= df.rollStd)
plt.show()

from statsmodels.tsa.stattools import adfuller
adfTest = adfuller(df['Monthly cola production'])
print(adfTest)
stats = pd.Series(adfTest[0:4],index = ['Test Statistic', 'p-value','#lags used','number of observations used'])
print(stats)

#H0 : time series is not stationary
#p-value >= 0.05: donot
for key, values in adfTest[4].items():
    print('criticality',key,":",values)


def test_stationarity(dataFrame, var):
    dataFrame['rollMean'] = dataFrame[var].rolling(window = 12).mean()
    dataFrame['rollStd'] = dataFrame[var].rolling(window = 12).std()
    from statsmodels.tsa.stattools import adfuller
    adfTest = adfuller(dataFrame[var])
    stats = pd.Series(adfTest[0:4], index = ['Test Statistic','p-value','#lags used','number of observations used'])
    print(stats)
    for key,values in adfTest[4].items():
        print('Criticality',key,":",values)

    plt.figure(figsize=(10,5))
    sns.lineplot(data = dataFrame, x = dataFrame.index, y = dataFrame['Monthly cola production'])
    sns.lineplot(data=dataFrame,x= dataFrame.index, y = dataFrame.rollMean)
    sns.lineplot(data = dataFrame, x= dataFrame.index, y= dataFrame.rollStd)
    plt.show()




air_df = df[['Monthly cola production']].copy()
print(air_df.head())

air_df['shift'] = air_df['Monthly cola production'].shift(1)
air_df['shiftDiff'] = air_df['Monthly cola production'] - air_df['shift']
print(air_df.head(20))

test_stationarity(air_df.dropna(),'shiftDiff')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_pacf(air_df['shift'].dropna(),lags=20)
plt.show()

plot_acf(air_df['shift'].dropna(),lags = 20)
plt.show()


train = air_df[:round(len(air_df) * 70/100)]
print(train.tail())

test = air_df[round(len(air_df)*70/100):]
print(test.head())


model = ARIMA(train['Monthly cola production'], order=(1,2,1))
model_fit = model.fit()
prediction = model_fit.predict(start = test.index[0],end = test.index[-1])
air_df['arimaPred'] = prediction
print(air_df.tail())

sns.lineplot(data = air_df,x = air_df.index,y ='Monthly cola production')
sns.lineplot(data = air_df,x =air_df.index, y= 'arimaPred')
plt.show()

air_df['diff12'] = air_df['Monthly cola production'].diff(12)

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train['Monthly cola production'], order=(1, 1, 1), seasonal_order=(3, 1, 3, 12))

model_fit = model.fit()
prediction = model_fit.predict(start=test.index[0], end=test.index[-1])
df['sarimaxPred'] = prediction
print(df.tail())
df.dropna()
print(df.head())
sns.lineplot(data =air_df,x = air_df.index,y = 'Monthly cola production' )
sns.lineplot(data =air_df,x = air_df.index,y = 'sarimaxPred' )
sns.lineplot(data =air_df,x = air_df.index,y = 'arimaPred' )
plt.show()



