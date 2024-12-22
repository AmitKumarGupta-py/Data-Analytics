import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



print("--------------------------------------------------")
print("Ques 1")
print("1.a\n")
df1 = pd.read_csv("201803-fordgobike-tripdata.csv")
print("\nDATA FRAME 1\n",df1.info())

df2 = pd.read_csv("201807-fordgobike-tripdata.csv")
print("\nDATA FRAME 2\n",df2.info())

df3 = pd.read_csv("201811-fordgobike-tripdata.csv")
print("\nDATA FRAME 3\n",df3.info())

df4 = pd.read_csv("201812-fordgobike-tripdata.csv")
print("\nDATA FRAME 4\n",df4.info())

combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

print("Combined DataFrame:\n",combined_df.info())
print(combined_df.columns)

# Convert to appropriate types
combined_df['start_time'] = pd.to_datetime(combined_df['start_time'])
combined_df['end_time'] = pd.to_datetime(combined_df['end_time'])


print("\nMemory Use:\n",combined_df.info(memory_usage='deep'))

average_duration = combined_df['duration_sec'].mean() / 60  # Convert to minutes
print(f'Average Trip Duration: {average_duration:.2f} minutes')

print("1.a\n")

plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
sns.histplot(combined_df['duration_sec'].sample(frac=0.05), bins=30, kde=True)
plt.title('Distribution of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.boxplot(x=combined_df['duration_sec'].sample(frac=0.05))
plt.title('Box Plot of Trip Durations')
plt.xlabel('Duration (seconds)')

plt.tight_layout()
plt.show()

print("1.b\n")

combined_df['start_time'] = pd.to_datetime(combined_df['start_time'])

combined_df['month'] = combined_df['start_time'].dt.month

avg_duration_per_month = combined_df.groupby('month')['duration_sec'].mean().reset_index()
avg_duration_per_month['duration_min'] = (avg_duration_per_month['duration_sec'] / 60)


print("1.c\n")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=avg_duration_per_month, x='month', y='duration_min')
plt.title('Average Trip Duration by Month')
plt.xlabel('Month')
plt.ylabel('Average Duration (minutes)')
plt.xticks(range(1, 13))
plt.grid()
plt.show()

print("1.d\n")


combined_df['start_time'] = pd.to_datetime(combined_df['start_time'])


combined_df['month'] = combined_df['start_time'].dt.month


def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


combined_df['season'] = combined_df['month'].apply(get_season)


avg_duration_by_season = combined_df.groupby('season')['duration_sec'].mean() / 60

plt.figure(figsize=(10, 6))
sns.barplot(x=avg_duration_by_season.index, y=avg_duration_by_season.values)
plt.title('Average Trip Duration by Season')
plt.xlabel('Season')
plt.ylabel('Average Duration (minutes)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

print("1.e\n")

short_trip_threshold = 600
combined_df['short_trip'] =combined_df['duration_sec'] < short_trip_threshold


plt.figure(figsize=(10, 6))
sns.countplot(data=combined_df, x='user_type', hue='short_trip')
plt.title('Short Trips by User Type')
plt.xlabel('User Type')
plt.ylabel('Count')
plt.legend(title='Short Trip', labels=['No', 'Yes'])
plt.show()

g = sns.FacetGrid(combined_df, col='member_gender', hue='short_trip', height=5, aspect=1)
g.map(sns.histplot, 'duration_sec', bins=30, stat='density', alpha=0.5)
g.add_legend()
g.set_axis_labels('Trip Duration (seconds)', 'Density')
g.set_titles(col_template="{col_name}")
plt.suptitle('Distribution of Trip Durations by Gender', fontsize=16)
plt.subplots_adjust(top=0.85)
plt.show()


short_trip_counts = combined_df.groupby(['user_type', 'short_trip']).size().unstack(fill_value=0)
print(short_trip_counts)

short_trip_proportions = short_trip_counts.div(short_trip_counts.sum(axis=1), axis=0)
print(short_trip_proportions)


print("--------------------------------------------------")
print("Ques 2")
print("2.a\n")

matrix_a = np.arange(0, 1.01, 0.01).reshape(101, 1)
matrix_a = np.tile(matrix_a, (1, 101))

print("\nMatrix A:\n")
print(matrix_a)

print("2.b\n")

array_b = np.linspace(0, 1, 20)

print("\nArray B:\n")
print(array_b)

print("2.c\n")

matrix_c = np.arange(1, 26).reshape(5, 5)

matrix_sum = np.sum(matrix_c)

matrix_std = np.std(matrix_c)

column_sums = np.sum(matrix_c, axis=0)

print("\nMatrix C:")
print(matrix_c)
print("\nSum of Matrix C:", matrix_sum)
print("Standard Deviation of Matrix C:", matrix_std)
print("Sum of each column in Matrix C:", column_sums)





print("--------------------------------------------------")
print("Ques 3")




x = np.arange(0, 100)
y = x * 2
z = x ** 2

fig = plt.figure(figsize=(10, 6))

ax1 = fig.add_axes([0, 0, 1, 1])
ax1.plot(x, z, label='x^2', color='blue')
ax1.set_title('Plot of x vs z (x^2)')
ax1.set_xlabel('x')
ax1.set_ylabel('z')
ax1.legend()
ax1.grid(True)

ax2 = fig.add_axes([0.2, 0.5, 0.2, 0.2])
ax2.plot(x, y, label='x*2', color='orange')
ax2.set_title('Plot of x vs y (x*2)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()
ax2.grid(True)

ax3 = fig.add_axes([0.2, 0.2, 0.4, 0.4])
ax3.plot(x, z, label='x^2', color='green')
ax3.set_title('x vs z in Smaller Axis')
ax3.set_xlabel('x')
ax3.set_ylabel('z')
ax3.legend()
ax3.grid(True)




plt.figure(figsize=(8, 4))
plt.plot(x, y, label='x*2', color='orange')
plt.title('Plot of x vs y (x*2) with limits')
plt.xlim(20, 22)
plt.ylim(30, 50)
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(y=30, color='r', linestyle='--')
plt.axhline(y=50, color='r', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()


print("--------------------------------------------------")
print("Ques 4")

df_w = pd.read_csv("walmart_purchase_data.csv")

print(df_w.columns)

print("\n4.a:\n")
average_purchase_price = df_w['Purchase Price'].mean()
print(f"Average Purchase Price: ${average_purchase_price:.2f}")

print("\n4.b:\n")
highest_purchase_price = df_w['Purchase Price'].max()
lowest_purchase_price = df_w['Purchase Price'].min()
print(f"Highest Purchase Price: ${highest_purchase_price:.2f}")
print(f"Lowest Purchase Price: ${lowest_purchase_price:.2f}")

print("\n4.c:\n")
english_speakers_count = df_w[df_w['Language'] == 'en'].shape[0]
print(f"Number of English speakers: {english_speakers_count}")

print("\n4.d:\n")

credit_card_number = 4926535242672853
email_of_cardholder = df_w[df_w['Credit Card'] == credit_card_number]['Email'].values
print(f"Email of the person with Credit Card {credit_card_number}: {email_of_cardholder[0] if email_of_cardholder else 'Not found'}")

print("\n4.e:\n")

purchase_price_lot_90wt = df_w[df_w['Lot'] == "90 WT"]['Purchase Price'].values
print(f"Purchase Price for Lot '90 WT': ${purchase_price_lot_90wt[0] if purchase_price_lot_90wt.size > 0 else 'Not found'}")

print("\n4.f:\n")
amex_high_value_count = df_w[(df_w['CC Provider'] == 'American Express') & (df_w['Purchase Price'] > 95)].shape[0]
print(f"Number of American Express users with purchase above $95: {amex_high_value_count}")

print("\n4.g:\n")
expiring_in_2025_count = df_w[df_w['CC Exp Date'].str.endswith('25')].shape[0]
print(f"Number of people with credit cards expiring in 2025: {expiring_in_2025_count}")

print("\n4.g:\n")
df_w['Email Domain'] = df_w['Email'].str.split('@').str[1]
top_email_providers = df_w['Email Domain'].value_counts().head(5)
print("\nTop 5 most popular email providers:")
print(top_email_providers)

print("\n4.i:\n")
top_job_titles = df_w['Job'].value_counts().head(5)
print("\nTop 5 most common Job Titles:")
print(top_job_titles)

print("\n4.j:\n")
morning_count = df_w[df_w['AM or PM'] == 'AM'].shape[0]
evening_count = df_w[df_w['AM or PM'] == 'PM'].shape[0]
print(f"Number of purchases in the morning: {morning_count}")
print(f"Number of purchases in the evening: {evening_count}")