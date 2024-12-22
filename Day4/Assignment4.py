import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("----------------------------------------------------------------")
print("Ques 1")

x = [10, 20, 30, 40, 50]
y = [10, 20, 30, 40, 50]

plt.figure(figsize=(10,6))
plt.plot(x,y,marker = 'o', color = 'b', linestyle = '-')
plt.title('Line Graph')
plt.xlabel('X label')
plt.ylabel('Y label')
plt.grid(True)
plt.show()

print("----------------------------------------------------------------")
print("Ques 2")

x = [1.0, 1.5, 2.0, 2.5, 3.0]
y = [2.0, 2.5, 4.0, 2.0, 1.0]

plt.figure(figsize=(10,6))
plt.plot(x,y,marker = 'o', color = 'b', linestyle = '-')
plt.title('Line Graph')
plt.xlabel('X label')
plt.ylabel('Y label')
plt.grid(True)
plt.show()

print("----------------------------------------------------------------")
print("Ques 3")

x = []
y = []

with open('test.txt', 'r') as file:
    for line in file:
        values = line.split()
        x.append(float(values[0]))
        y.append(float(values[1]))


plt.figure(figsize=(10,6))
plt.plot(x,y,marker = 'o', color = 'b', linestyle = '-')
plt.title('Line Graph')
plt.xlabel('X label')
plt.ylabel('Y label')
plt.grid(True)
plt.show()

print("----------------------------------------------------------------")
print("Ques 4")

df = pd.read_csv('fdata.csv')

df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y')

df.set_index('Date', inplace=True)

plt.figure(figsize=(12, 6))

plt.plot(df.index, df['Open'], marker='o', label='Open', color='blue')
plt.plot(df.index, df['High'], marker='o', label='High', color='green')
plt.plot(df.index, df['Low'], marker='o', label='Low', color='red')
plt.plot(df.index, df['Close'], marker='o', label='Close', color='orange')


plt.title('Financial Data of Alphabet Inc. (Oct 3, 2016 - Oct 7, 2016)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()


plt.tight_layout()
plt.show()


print("----------------------------------------------------------------")
print("Ques 5")

x = [1.0, 1.5, 2.0, 2.5, 3.0]
y = [2.0, 2.5, 4.0, 2.0, 1.0]

plt.figure(figsize=(10,6))
plt.plot(x,y,marker = 'o', color = 'b', linestyle = '-',label = 'line1')
plt.plot(y,x,marker = 'x', color = 'g', linestyle = '-',label = 'line2')
plt.title('Line Graph')
plt.xlabel('X label')
plt.ylabel('Y label')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


print("----------------------------------------------------------------")
print("Ques 6")


x = [1.0, 1.5, 2.0, 2.5, 3.0]
y1 = [2.0, 2.5, 4.0, 2.0, 1.0]
y2 = [3.0, 3.6, 4.5,5.0,6.0]

plt.figure(figsize=(10,6))
plt.plot(x,y1,marker = 'o', color = 'b', linestyle = '-',label = 'line1',linewidth=2)
plt.plot(y1,x,marker = 'x', color = 'g', linestyle = '-',label = 'line2',linewidth=1.5)

plt.plot(x,y2,marker = 'x', color = 'r', linestyle = '-',label = 'line3',linewidth=3)
plt.title('Line Graph')
plt.xlabel('X label')
plt.ylabel('Y label')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("----------------------------------------------------------------")
print("Ques 7")


x = [1.0, 1.5, 2.0, 2.5, 3.0]
y1 = [2.0, 2.5, 4.0, 2.0, 1.0]
y2 = [3.0, 3.6, 4.5,5.0,6.0]

plt.figure(figsize=(10,6))
plt.plot(x,y1, color = 'b', linestyle = '-',label = 'line1',linewidth=2)
plt.plot(y1,x, color = 'g', linestyle = '--',label = 'line2',linewidth=1.5)

plt.plot(x,y2, color = 'r', linestyle = '-',label = 'line3',linewidth=3)
plt.title('Line Graph')
plt.xlabel('X label')
plt.ylabel('Y label')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("----------------------------------------------------------------")
print("Ques 8")


x = [1.0, 1.5, 2.0, 2.5, 3.0]
y1 = [2.0, 2.5, 4.0, 2.0, 1.0]
y2 = [3.0, 3.6, 4.5,5.0,6.0]

plt.figure(figsize=(10,6))
plt.plot(x,y1,marker = 'o', color = 'g', linestyle = '-',label = 'line1',linewidth=2)
plt.plot(y1,x,marker = 'x', color = 'r', linestyle = '--',label = 'line2',linewidth=1.5)

plt.plot(x,y2,marker = 's', color = 'b', linestyle = '-',label = 'line3',linewidth=3)
plt.title('Line Graph')
plt.xlabel('X label')
plt.ylabel('Y label')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


print("----------------------------------------------------------------")
print("Ques 9")


x = [1.0, 1.5, 2.0, 2.5, 3.0]
y1 = [2.0, 2.5, 4.0, 2.0, 1.0]
y2 = [3.0, 3.6, 4.5,5.0,6.0]

plt.figure(figsize=(10,6))
plt.plot(x,y1,marker = 'o', color = 'b', linestyle = '-',label = 'line1',linewidth=2)
plt.plot(y1,x,marker = 'x', color = 'g', linestyle = '--',label = 'line2',linewidth=1.5)

plt.plot(x,y2,marker = 's', color = 'r', linestyle = '-',label = 'line3',linewidth=3)
plt.title('Line Graph')
plt.xlabel('X label')
plt.ylabel('Y label')
current_limits = plt.axis()
print(f"Current axis limits: {current_limits}")

new_x_limits = (0, 6)
new_y_limits = (0, 12)
plt.axis(new_x_limits + new_y_limits)

print(f"New axis limits: {plt.axis()}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("----------------------------------------------------------------")
print("Ques 10")

x_positions = [1, 2, 3, 4, 5, 6,8,3,4,5,2]
y_positions = [2, 3, 5, 1, 4, 6,2,4,5,1,6]

plt.figure(figsize= (8,6))
plt.scatter(x_positions, y_positions, color='blue', marker='o',alpha = 0.6)

plt.title("Scatter Plot of Positions")
plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.grid(False)
plt.show()

print("----------------------------------------------------------------")
print("Ques 11")

x = np.linspace(0,5,100)
y = np.exp(x)
y1 = np.exp(x+5)
y2 = np.exp(x+2)


plt.figure(figsize=(10, 6))

plt.scatter(x, y, color='red', label='True Exponential Curve', alpha=0.6)
plt.scatter(x, y1, color='blue', label='True Exponential Curve', alpha = 0.6)
plt.scatter(x, y2, color='green', label='True Exponential Curve', alpha = 0.6)

plt.title("Scatter Plot of Positions")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid()
plt.ylim(bottom = 0)

plt.show()


print("----------------------------------------------------------------")
print("Ques 12")


x = np.linspace(0, 10, 100)

y_curve = np.sin(x)

x_scatter = np.array([1, 3, 5, 7, 9])
y_scatter = np.sin(x_scatter)

plt.figure(figsize=(10, 6))

plt.plot(x, y_curve, label='Sine Curve', color='blue', linewidth=2)

plt.scatter(x_scatter, y_scatter, color='red', label='Specific Points', s=100, marker='o')

plt.axhline(0, color='black', linewidth=0.5, linestyle='--')

plt.title("Multiple Chart Types on a Single Set of Axes")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.legend()

plt.grid()
plt.show()


print("----------------------------------------------------------------")
print("Ques 13")

data = {
    'Date': ['03-10-16', '04-10-16', '05-10-16', '06-10-16', '07-10-16'],
    'Close': [772.559998, 776.429993, 776.469971, 776.859985, 775.080017]
}

df = pd.DataFrame(data)

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y')

plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'], marker='o', label='Closing Value', color='blue')

plt.grid(color='red', linestyle='-', linewidth=0.5)


plt.title("Closing Value of Alphabet Inc. (Oct 3 - Oct 7, 2016)")
plt.xlabel("Date")
plt.ylabel("Closing Value (USD)")


plt.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("----------------------------------------------------------------")
print("Ques 14")

data = {
    'Date': ['03-10-16', '04-10-16', '05-10-16', '06-10-16', '07-10-16'],
    'Close': [772.559998, 776.429993, 776.469971, 776.859985, 775.080017]
}

df = pd.DataFrame(data)

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y')


plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'], marker='o', label='Closing Value', color='red')


plt.grid(which='both', color='red', linestyle='-', linewidth=0.5)
plt.minorticks_on()
plt.grid(which='minor', color='lightblue', linestyle=':', linewidth=0.5)

plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

plt.title("Closing Value of Alphabet Inc. (Oct 3 - Oct 7, 2016)")
plt.xlabel("Date")
plt.ylabel("Closing Value (USD)")

plt.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


print("----------------------------------------------------------------")
print("Ques 15")



fig, axs = plt.subplots(2, 2, figsize=(12, 8))

plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)


plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)



plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)



plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)


plt.tight_layout()
plt.show()