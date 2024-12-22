import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches
import seaborn as sns
from IPython.core.pylabtools import figsize
from matplotlib.patches import Rectangle

print("------------------------------------------------------------")
print("Ques 1")

Programming_languages = ['Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C++']
Popularity = [22.2, 17.6, 8.8, 8, 7.7, 6.7]

sns.set(style = 'whitegrid')
plt.figure(figsize=(8,6))
sns.barplot(x= Programming_languages, y= Popularity )
plt.title('Popularity of Programming Language',fontsize = 12)
plt.xlabel('Languages')
plt.ylabel('Popularity')
plt.show()

print("------------------------------------------------------------")
print("Ques 2")

Programming_languages = ['Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C++']
Popularity = [22.2, 17.6, 8.8, 8, 7.7, 6.7]

sns.set(style = 'whitegrid')
plt.figure(figsize=(8,6))
sns.barplot(y= Programming_languages, x= Popularity )
plt.title('Popularity of Programming Language',fontsize = 12)
plt.xlabel('Languages')
plt.ylabel('Popularity')
plt.show()

print("------------------------------------------------------------")
print("Ques 3")

Programming_languages = ['Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C++']
Popularity = [22.2, 17.6, 8.8, 8, 7.7, 6.7]

sns.set(style = 'whitegrid')
plt.figure(figsize=(8,6))
sns.barplot(x= Programming_languages, y= Popularity )

plt.gca().invert_yaxis()
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.title('Popularity of Programming Language',fontsize = 12)
plt.xlabel('Languages')
plt.ylabel('Popularity')
plt.legend()
plt.show()


print("------------------------------------------------------------")
print("Ques 4")

data = {'Programming_languages' : ['Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C++'],
'Popularity' : [22.2, 17.6, 8.8, 8, 7.7, 6.7]
        }
df = pd.DataFrame(data)
sns.set(style = 'whitegrid')
plt.figure(figsize=(8,6))
sns.barplot(x= 'Programming_languages', y= 'Popularity',data = df,palette='viridis')
plt.title('Popularity of Programming Language',fontsize = 12)
plt.xlabel('Languages')
plt.ylabel('Popularity')
plt.show()

print("------------------------------------------------------------")
print("Ques 5")


data = {'Programming_languages' : ['Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C++'],
'Popularity' : [22.2, 17.6, 8.8, 8, 7.7, 6.7]
        }
df = pd.DataFrame(data)
sns.set(style = 'whitegrid')
plt.figure(figsize=(10,8))
sns.barplot(x= 'Programming_languages', y= 'Popularity',data = df,palette='viridis')
for index,row in df.iterrows():
    plt.text(index,row['Popularity']+1,f'{row['Popularity']}',va = 'bottom',ha = 'center')
plt.title('Popularity of Programming Language',fontsize = 12)
plt.xlabel('Languages')
plt.ylabel('Popularity')
plt.show()


print("------------------------------------------------------------")
print("Ques 6")

data = {'Programming_languages' : ['Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C++'],
'Popularity' : [22.2, 17.6, 8.8, 8, 7.7, 6.7]
        }
df = pd.DataFrame(data)
sns.set(style = 'whitegrid')
plt.figure(figsize=(8,6))
sns.barplot(x= 'Programming_languages', y= 'Popularity',data = df,palette='viridis',edgecolor  = 'blue' )
plt.title('Popularity of Programming Language',fontsize = 12)
plt.xlabel('Languages')
plt.ylabel('Popularity')
plt.show()

print("------------------------------------------------------------")
print("Ques 7")

data = {'Programming_languages' : ['Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C++'],
'Popularity' : [22.2, 17.6, 8.8, 8, 7.7, 6.7]
        }
df = pd.DataFrame(data)

positions = [0,1,4,5,6,8]

plt.figure(figsize=(8,6))
plt.bar(x= positions, height = df['Popularity'],edgecolor  = 'blue',zorder = 2)
plt.title('Popularity of Programming Language',fontsize = 12)
plt.xticks(positions,df['Programming_languages'])
plt.xlabel('Languages')
plt.ylabel('Popularity')
plt.show()


print("------------------------------------------------------------")
print("Ques 8")

data = {'Programming_languages' : ['Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C++'],
'Popularity' : [22.2, 17.6, 8.8, 8, 7.7, 6.7]
        }
df = pd.DataFrame(data)

positions = [0,1,4,5,6,8]
width_position=[0.8,0.2,0.4,0.5,0.1,0.3]
plt.figure(figsize=(8,6))
plt.bar(x= positions, height = df['Popularity'],width = width_position,edgecolor  = 'blue',zorder = 2)
plt.title('Popularity of Programming Language',fontsize = 12)
plt.xticks(positions,df['Programming_languages'])
plt.xlabel('Languages')
plt.ylabel('Popularity')
plt.show()


print("------------------------------------------------------------")
print("Ques 9")

data = {'Programming_languages' : ['Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C++'],
'Popularity' : [22.2, 17.6, 8.8, 8, 7.7, 6.7]
        }
df = pd.DataFrame(data)
sns.set(style = 'whitegrid')
plt.figure(figsize=(8,6))
sns.barplot(x= 'Programming_languages', y= 'Popularity',data = df,palette='viridis')
plt.title('Popularity of Programming Language',fontsize = 12)
plt.xlabel('Languages')
plt.ylabel('Popularity')
plt.subplots_adjust(bottom=0.5)
plt.show()

print("------------------------------------------------------------")
print("Ques 10")

data = {
    'group':['G1','G2','G3','G4','G5'],
    'men': [22,30,35,35,26],
    'women':[25,32,30,35,29]
}
df = pd.DataFrame(data)
bar_width = 0.35
positions = np.arange(len(df['group']))
plt.figure(figsize=(10, 6))
plt.bar(positions - bar_width/2, df['men'],width= bar_width, label='Men', edgecolor='blue')
plt.bar( positions + bar_width/2,df['women'],width = bar_width, label='Women', edgecolor='pink')


plt.title('Scores by Group and Gender', fontsize=14)
plt.xlabel('Groups', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.xticks(positions, df['group'])
plt.legend()

plt.tight_layout()
plt.show()

print("------------------------------------------------------------")
print("Ques 11")

data = {
    'a' : [2,4,8,5,7,6],
    'b': [4,2,3,4,2,6],
    'c': [6,4,7,4,7,8],
    'd': [8,2,6,4,8,6],
    'e': [10,2,4,3,3,2]
}
df = pd.DataFrame(data)

positions = range(len(df))

bar_width = 0.15

for i, column in enumerate(['a','b', 'c', 'd', 'e']):
    plt.bar([pos + i * bar_width for pos in positions],
            df[column],
            width=bar_width,
            label=column)

plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Bar Plot from DataFrame')
plt.xticks([pos + bar_width * 1.5 for pos in positions], df['a'])
plt.legend()

plt.tight_layout()
plt.show()


print("------------------------------------------------------------")
print("Ques 12")

data= {
    'Mean_velocity': [0.2474, 0.1235, 0.1737, 0.1824],
'Standard_deviation': [0.3314, 0.2278, 0.2836, 0.2645]
}

df = pd.DataFrame(data)

plt.figure(figsize=(8,6))
sns.barplot(data=df, x='Mean_velocity', y='Standard_deviation',capsize=0.1,palette = 'viridis',ci = None)

for i in range(len(df)):
    plt.errorbar(i, df['Mean_velocity'][i],
                 yerr=df['Standard_deviation'][i],
                 fmt='none', color='black', capsize=5)

plt.xlabel('Velocity')
plt.ylabel('Standard_deviation')
plt.title('Mean Velocity with Error bars')
plt.legend()
plt.tight_layout()
plt.show()


print("------------------------------------------------------------")
print("Ques 13")

data= {
    'Mean_velocity': [0.2474, 0.1235, 0.1737, 0.1824],
'Standard_deviation': [0.3314, 0.2278, 0.2836, 0.2645],
    'men': [22,30,35,35]
}

df = pd.DataFrame(data)

plt.figure(figsize=(8,6))
sns.barplot(data=df, x='Mean_velocity', y='Standard_deviation',capsize=0.1,palette = 'viridis',ci = None)

for i in range(len(df)):
    plt.errorbar(i, df['Mean_velocity'][i],
                 yerr=df['Standard_deviation'][i],
                 fmt='none', color='black', capsize=5)

for index,row in df.iterrows():
    plt.text(index,row['Mean_velocity'],f'{row['men']}',va = 'bottom',ha = 'center')

plt.xlabel('Velocity')
plt.ylabel('Standard_deviation')
plt.title('Mean Velocity with Error bars')
plt.legend()
plt.tight_layout()
plt.show()

print("------------------------------------------------------------")
print("Ques 14")

means_men = np.array([22, 30, 35, 35, 26])
means_women = np.array([25, 32, 30, 35, 29])
std_dev_men = np.array([4, 3, 4, 1, 5])
std_dev_women = np.array([3, 5, 2, 3, 3])

labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5']
x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(10, 6))

bars_men = ax.bar(x, means_men, yerr=std_dev_men, capsize=5, color='lightblue', label='Men', alpha=0.7)

bars_women = ax.bar(x, means_women, yerr=std_dev_women, capsize=5, bottom=means_men, color='lightcoral', label='Women', alpha=0.7)

ax.set_xlabel('Groups')
ax.set_ylabel('Mean Values')
ax.set_title('Stacked Bar Plot with Error Bars')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


plt.tight_layout()
plt.show()


print("------------------------------------------------------------")
print("Ques 15")

languages = [['Language', 'Science', 'Math'],
             ['Science', 'Math', 'Language'],
             ['Math', 'Language', 'Science']]
numbers = [{'Language': 75, 'Science': 88, 'Math': 96},
           {'Language': 71, 'Science': 95, 'Math': 92},
           {'Language': 75, 'Science': 90, 'Math': 89}]


labels = ['Student 1', 'Student 2', 'Student 3']
data = np.zeros((len(labels), 3))
colors = ['lightblue', 'lightcoral', 'lightgreen']


for i, lang_order in enumerate(languages):
    for j, lang in enumerate(lang_order):
        data[i, j] = numbers[i][lang]


fig, ax = plt.subplots(figsize=(10, 6))


for i in range(data.shape[1]):
    if i == 0:
        ax.barh(labels, data[:, i], color=colors[i], label=languages[0][i])
    else:
        ax.barh(labels, data[:, i], left=np.sum(data[:, :i], axis=1), color=colors[i], label=languages[0][i])


ax.set_xlabel('Scores')
ax.set_title('Horizontal Stacked Bar Chart with Different Colors')
ax.legend()


plt.tight_layout()
plt.show()

print("------------------------------------------------------------")
print("Ques 16")

people = ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8')
segments = 4

data = [
    [3.40022085, 7.70632498, 6.4097905, 10.51648577, 7.5330039, 7.1123587, 12.77792868, 3.44773477],
    [11.24811149, 5.03778215, 6.65808464, 12.32220677, 7.45964195, 6.79685302, 7.24578743, 3.69371847],
    [3.94253354, 4.74763549, 11.73529246, 4.6465543, 12.9952182, 4.63832778, 11.16849999, 8.56883433],
    [4.24409799, 12.71746612, 11.3772169, 9.00514257, 10.47084185, 10.97567589, 3.98287652, 8.80552122]
]

data = np.array(data)

fig, ax = plt.subplots(figsize=(10, 6))


bottom = np.zeros(len(people))

colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']

for i in range(segments):
    ax.bar(people, data[i], bottom=bottom, color=colors[i], label=f'Segment {i + 1}')

    for j in range(len(people)):
        ax.text(j, bottom[j] + data[i][j] / 2, f'{data[i][j]:.1f}',
                ha='center', va='center', fontsize=9, color='black')
    bottom += data[i]


ax.set_xlabel('Groups')
ax.set_ylabel('Values')
ax.set_title('Stacked Bar Plot with Labels on Each Section')
ax.legend()


plt.tight_layout()
plt.show()


print("------------------------------------------------------------")
print("Ques 17")

def add_texture(ax, rect, pattern):

    texture = Rectangle((rect.get_x(), rect.get_y()), rect.get_width(), rect.get_height(),
                        hatch=pattern, facecolor='none', edgecolor='black')
    ax.add_patch(texture)


labels = ['A', 'B', 'C', 'D']
men_means = [20, 35, 30, 35]
women_means = [25, 32, 34, 20]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()


men_bars = ax.bar(x, men_means, width, label='Men', color='lightgrey')

women_bars = ax.bar(x, women_means, width, bottom=men_means, label='Women', color='whitesmoke')


for bar in men_bars:
    add_texture(ax, bar, '/')

for bar in women_bars:
    add_texture(ax, bar, '\\')


ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.show()