import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
from babel.dates import date_
from numba.cuda.printimpl import print_item
from scipy.signal import freqs
from sympy.stats import given

print("-------------------------------------------------------------------")
print("Code 1")

date_time_object = pd.to_datetime('Jan 15 2012 ')
print("a) Datetime object for Jan 15 2012:\n",date_time_object)

date_time_object1 = pd.to_datetime('Jan 15 2012 9:20 pm')
print("\nb) Specific date and time of 9:20 pm:\n",date_time_object1)

date_time_object2 = pd.to_datetime('Jan 15 2012 9:20 pm').tz_localize('Asia/Kolkata')
print("\nc) Local date and time:\n",date_time_object2)

date_time_object3 = pd.to_datetime('Jan 15 2012 ').date()
print("\nd) A date without time:\n",date_time_object3)

date_time_object4 = datetime.now().date()
print("\ne) Current date:\n",date_time_object4)

date_time_object5 = pd.to_datetime('9:20 pm').time()
print("\nf) Time from a datetime:\n",date_time_object5)

date_time_object6 = datetime.now().time()
print("\ng) Current local time:\n",date_time_object6)

print("-------------------------------------------------------------------")
print("Code 2")

specific_datetime = pd.Timestamp('2023-10-21').date()
print("\na) Specific date:\n", specific_datetime)

specific_datetime1 = pd.Timestamp('Jan 15 2012 9:20 pm')
print("\nb) date and time using timestamp:\n",specific_datetime1)

#2.c
current_date = pd.Timestamp.now().date()

specific_time = pd.Timestamp('15:30:00').time()

combined_datetime = pd.Timestamp.combine(current_date, specific_time)

print("\nc) a time adds in the current local date using timestamp:\n",combined_datetime)

#2.d

specific_time1 = pd.Timestamp.now()
print("\nd) current date and time using timestamp:\n",specific_time1)


print("-------------------------------------------------------------------")
print("Code 3")


year = 2023
month = 10
day = 21

date_from_components = pd.Timestamp(year, month, day)

date_string = '2023-10-21'
date_from_string = pd.to_datetime(date_string)


print("Date from components (year, month, day):", date_from_components)
print("Date from string format:", date_from_string)

print("-------------------------------------------------------------------")
print("Code 4")


def print_days_around_date(date_str):

    date = pd.to_datetime(date_str)


    day_before = date - pd.Timedelta(days=1)
    day_after = date + pd.Timedelta(days=1)

    print(f"Specified Date: {date.date()}")
    print(f"Day Before: {day_before.date()}")
    print(f"Day After: {day_after.date()}")



def days_between_dates(start_date_str, end_date_str):

    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)


    days_diff = (end_date - start_date).days

    print(f"Days between {start_date.date()} and {end_date.date()}: {days_diff} days")



print_days_around_date('2024-10-21')
days_between_dates('2024-10-01', '2024-10-21')


print("-------------------------------------------------------------------")
print("Code 5")

date_range = pd.date_range(start='2024-09-14', periods=5, freq='D')

index = pd.MultiIndex.from_product([date_range, ['A', 'B']], names=['Date', 'Label'])

random_values = np.random.rand(len(index))

time_series = pd.Series(random_values, index=index)

print(time_series)

print(f"\nType of the index: {type(time_series.index)}\n")

print("-------------------------------------------------------------------")
print("Code 6")

list_dates = ['2024 09 14','2024 09 11','2024 10 1','2023 09 14']

date_series = pd.to_datetime(list_dates)

print(date_series)

values = [10,20,15,30]

time_series1 = pd.Series(values,index = date_series)
print(time_series1)

print(f"Type of the index: {type(time_series1.index)}")


print("-------------------------------------------------------------------")
print("Code 7")

date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='MS')

data = [i + 1 for i in range(len(date_range))]

time_series = pd.Series(data, index=date_range)

print("Original Time Series:")
print(time_series)

year_2024_data = time_series[time_series.index.year == 2024]

print("\nData for the Year 2024:")
print(year_2024_data)

start_date = '2024-03-01'
end_date = '2024-06-30'
filtered_data = time_series[start_date:end_date]

print(f"\nData between {start_date} and {end_date}:")
print(filtered_data)

print("-------------------------------------------------------------------")
print("Code 8")

start_date = '2024-09-11'
num_periods = 10

date_range = pd.date_range(start=start_date, periods=num_periods)

print(date_range)

print("-------------------------------------------------------------------")
print("Code 9")

date_range = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')

max_timestamp = date_range.max()
min_timestamp = date_range.min()

max_index = date_range.get_loc(max_timestamp)
min_index = date_range.get_loc(min_timestamp)

print("Date Range:")
print(date_range)
print("\nMaximum Timestamp:", max_timestamp)
print("Index of Maximum Timestamp:", max_index)
print("\nMinimum Timestamp:", min_timestamp)
print("Index of Minimum Timestamp:", min_index)

print("-------------------------------------------------------------------")
print("Code 10")

start_date = '2024-01-01'
num_periods = 6

date_range = pd.date_range(start=start_date, periods=num_periods, freq='3MS')

time_series = pd.Series(range(num_periods), index=date_range)

print("Time Series with Three Months Frequency:")
print(time_series)

print("-------------------------------------------------------------------")
print("Code 11")

num_durations = 10

durations = pd.to_timedelta(range(num_durations), unit='h')

print("Sequence of Durations Increasing by an Hour:")
print(durations)

print("-------------------------------------------------------------------")
print("Code 12")

data = {
    'Year': [2024, 2023, 2022, 2021],
    'DayOfYear': [1, 100, 200, 365]
}

df = pd.DataFrame(data)

df['Date'] = pd.to_datetime(df['Year'].astype(str) + df['DayOfYear'].astype(str), format='%Y%j')

print("DataFrame with Year, DayOfYear, and Date:")
print(df)


print("-------------------------------------------------------------------")
print("Code 13")

data = {
    'Year': [2024, 2023, 2022, 2021],
    'Month': [1, 5, 8, 12],
    'Day': [15, 10, 25, 1]
}

df = pd.DataFrame(data)

df['Timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

print("DataFrame with Timestamps:")
print(df)

data_strings = {
    'DateString': ['2024-01-15', '2023-05-10', '2022-08-25', '2021-12-01']
}

df_strings = pd.DataFrame(data_strings)

df_strings['Timestamp'] = pd.to_datetime(df_strings['DateString'])

print("\nDataFrame from String Dates with Timestamps:")
print(df_strings)

print("-------------------------------------------------------------------")
print("Code 14")

def is_business_day(date):

    date = pd.to_datetime(date)
    return date.dayofweek < 5

dates_to_check = ['2024-01-01', '2024-01-06', '2024-01-08', '2024-01-13']

results = pd.DataFrame({'Date': dates_to_check})

results['IsBusinessDay'] = results['Date'].apply(is_business_day)

print("Business Day Check:")
print(results)


print("-------------------------------------------------------------------")
print("Code 15")

year = 2024
date_range = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='BM')

print("Last Working Days of Each Month in", year, ":")
print(date_range)

days_of_week = date_range.dayofweek
days_of_week_name = date_range.day_name()

results = pd.DataFrame({
    'Date': date_range,
    'DayOfWeekInt': days_of_week,
    'DayOfWeekName': days_of_week_name
})
print("Last Business Days and Their Corresponding Days of the Week:")
print(results)

print("-------------------------------------------------------------------")
print("Code 16")

start_date = '2024-01-01 08:00'
num_periods = 10

time_series = pd.date_range(start=start_date, periods=num_periods, freq='30T')

print("Time Series Combining Hour and Minute:")
print(time_series)

print("-------------------------------------------------------------------")
print("Code 17")

unix_time = 1683024000
utc_timestamp = pd.to_datetime(unix_time, unit='s', utc=True)
time_zone = 'America/New_York'
localized_timestamp = utc_timestamp.tz_convert(time_zone)

print("UTC Timestamp:", utc_timestamp)
print("Localized Timestamp in", time_zone + ":", localized_timestamp)

print("-------------------------------------------------------------------")
print("Code 18")

start_date = '2024-01-01'
end_date = '2024-01-10'
freq = 'D'

date_range = pd.date_range(start=start_date, end=end_date, freq=freq).tz_localize('Asia/Kolkata')

time_series = pd.Series(range(len(date_range)), index=date_range)

print("Time Series Object with Time Zone:")
print(time_series)

print("-------------------------------------------------------------------")
print("Code 19")

start_date = '2024-01-01'
end_date = '2024-01-10'
freq = 'D'

date_range = pd.date_range(start=start_date, end=end_date, freq=freq).tz_localize('America/New_York')

time_series = pd.Series(range(len(date_range)), index=date_range)

print("Original Time Series with Time Zone:")
print(time_series)

naive_time_series = time_series.tz_localize(None)

print("\nTime Series without Time Zone Information:")
print(naive_time_series)

print("-------------------------------------------------------------------")
print("Code 20")

timestamp1 = pd.to_datetime('2024-10-21 10:00:00').tz_localize('UTC')
timestamp2 = pd.to_datetime('2024-10-21 08:00:00').tz_localize('America/New_York')

timestamp2_utc = timestamp2.tz_convert('UTC')

time_difference = timestamp1 - timestamp2_utc

print(f"Timestamp 1 (UTC): {timestamp1}")
print(f"Timestamp 2 (UTC): {timestamp2_utc}")
print(f"Time Difference: {time_difference}")

print("-------------------------------------------------------------------")
print("Code 21")

start_date = '2024-01-01'
end_date = '2024-12-31'

thursdays = pd.date_range(start=start_date, end=end_date, freq='W-THU')

print("All Thursdays between", start_date, "and", end_date, ":")
print(thursdays)


print("-------------------------------------------------------------------")
print("Code 22")

year = 2024

quarter_starts = pd.date_range(start=f'{year}-01-01', periods=4, freq='Q')
quarter_ends = pd.date_range(start=f'{year}-01-01', periods=4, freq='Q-DEC')

print(f"Quarterly Begin Dates for {year}:")
print(quarter_starts)

print(f"\nQuarterly End Dates for {year}:")
print(quarter_ends)

print("-------------------------------------------------------------------")
print("Code 23")

start_date = '2024-01-01'
end_date = '2024-01-10'

daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
print("Daily Dates:")
print(daily_dates)

monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')

start_time = '0 days'
end_time = '10 days'

daily_time_spans = pd.timedelta_range(start=start_time, end=end_time, freq='D')
print("\nDaily Time Spans:")
print(daily_time_spans)


hourly_time_spans = pd.timedelta_range(start=start_time, end=end_time, freq='H')
print("\nHourly Time Spans:")
print(hourly_time_spans)


print("-------------------------------------------------------------------")
print("Code 24")

start_date = '2024-01-01'
end_date = '2024-01-07'

daily_range = pd.date_range(start=start_date, end=end_date, freq='D')

combined_time_series = []

for day in daily_range:
    intraday_range = pd.date_range(start=day, end=day + pd.Timedelta(hours=23, minutes=59), freq='H')

    combined_time_series.extend(intraday_range)

final_time_series = pd.DatetimeIndex(combined_time_series)

time_series_df = pd.DataFrame(final_time_series, columns=['Datetime'])

print("\ngenerate time series combining day and intraday offsets intervals:\n",time_series_df)


print("-------------------------------------------------------------------")
print("Code 25")

specified_date = '2024-01-01'

date = pd.to_datetime(specified_date)

day_name = date.day_name()
print(f"Day name for {specified_date}: {day_name}")

date_plus_2_days = date + pd.DateOffset(days=2)
print(f"Date after adding 2 days: {date_plus_2_days.date()}")

date_plus_1_business_day = date + pd.offsets.BDay(1)
print(f"Date after adding 1 business day: {date_plus_1_business_day.date()}")

print("-------------------------------------------------------------------")
print("Code 26")

epoch_times = [1609459200, 1609545600, 1609632000]

timestamps = [pd.to_datetime(epoch, unit='s') for epoch in epoch_times]
print("Timestamps:")
for epoch, timestamp in zip(epoch_times, timestamps):
    print(f"Epoch: {epoch} -> Timestamp: {timestamp}")

datetime_index = pd.to_datetime(epoch_times, unit='s')
print("\nDatetimeIndex:")
print(datetime_index)

print("-------------------------------------------------------------------")
print("Code 27")

specified_date = '2024-01-01'

date = pd.to_datetime(specified_date)

one_business_day = date + pd.offsets.BDay(1)
two_business_days = date + pd.offsets.BDay(2)
three_business_days = date + pd.offsets.BDay(3)

print(f"Specified date: {date.date()}")
print(f"One business day from {date.date()}: {one_business_day.date()}")
print(f"Two business days from {date.date()}: {two_business_days.date()}")
print(f"Three business days from {date.date()}: {three_business_days.date()}")

next_business_month_end = (date + pd.offsets.BMonthEnd(1))
print(f"Next business month end from {date.date()}: {next_business_month_end.date()}")


print("-------------------------------------------------------------------")
print("Code 28")

year = 2024

monthly_boundaries = pd.period_range(start=f'{year}-01', end=f'{year}-12', freq='M')

print("Monthly Boundaries Period Index:")
print(monthly_boundaries)

print("\nStart and End times for each period:")
for period in monthly_boundaries:
    start_time = period.start_time
    end_time = period.end_time
    print(f"Period: {period}, Start: {start_time}, End: {end_time}")

print("-------------------------------------------------------------------")
print("Code 29")

periods_2029_2031 = pd.period_range(start='2029-01', end='2031-12', freq='M')

data = pd.Series(range(len(periods_2029_2031)), index=periods_2029_2031)

print("Series with PeriodIndex representing months in 2029 and 2031:")
print(data)

print("\nValues for all periods in 2030:")
values_2030 = data['2030']
print(values_2030)

print("-------------------------------------------------------------------")
print("Code 30")

start_date = '2024-01-01'
end_date = '2024-12-31'

date_range = pd.date_range(start=start_date, end=end_date)

calendar = USFederalHolidayCalendar()

holidays = calendar.holidays(start=start_date, end=end_date)

print(f"US Federal Holidays between {start_date} and {end_date}:")
print(holidays)

print("-------------------------------------------------------------------")
print("Code 31")

monthly_periods = pd.period_range(start='2024-01', end='2024-12', freq='M')

print("Monthly Time Periods for 2024:")
print(monthly_periods)

print("\nNames in the current local scope:")
print(locals())

print("-------------------------------------------------------------------")
print("Code 32")

specified_year = 2024

yearly_period = pd.Period(year=specified_year, freq='Y')

print(f"Yearly Time Period for {specified_year}:")
print(f"Period: {yearly_period}")
print(f"Start Time: {yearly_period.start_time}")
print(f"End Time: {yearly_period.end_time}")
print(f"Frequency: {yearly_period.freq}")
print(f"Year: {yearly_period.year}")
print(f"Month: {yearly_period.month}")
print(f"Day: {yearly_period.day}")