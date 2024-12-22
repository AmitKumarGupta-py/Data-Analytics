import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter

df =pd.read_csv('weight-height.csv')
print(df.head())

#separate data based on gender

df_male = df[df['Gender'] == 'Male']
df_female = df[df['Gender'] == 'Female']

#calculate correlation
corr_overall = df[['Height','Weight']].corr()
corr_male = df_male[['Height','Weight']].corr()
corr_female = df_female[['Height','Weight']].corr()

#calculate covariance


cov_overall = df[['Height','Weight']].cov()
cov_male = df_male[['Height','Weight']].cov()
cov_female = df_female[['Height','Weight']].cov()


print("\nOverall Correlation b/w Height and weight:\n")
print(corr_overall)

print("\nMale Correlation b/w Height and weight:\n")

print(corr_male)
print("\nfemale Correlation b/w Height and weight:\n")

print(corr_female)

print("\nOverall Cavariance b/w Height and weight:\n")
print(cov_overall)

print("\nMale Cavariance b/w Height and weight:\n")

print(cov_male)
print("\nfemale cavariance b/w Height and weight:\n")

print(cov_female)

print("\n",df.info())

#create  plot

fig, axes = plt.subplots(1,3, figsize = (18,6))

#Overall DATA plot

axes[0].scatter(df['Height'],df['Weight'],color = 'blue', alpha = 0.5)
axes[0].set_title(f'Overall Correlation:{corr_overall.loc["Height","Weight"]: .2f}')
axes[0].set_xlabel('Height')
axes[0].set_ylabel('Weight')
axes[0].grid(True)

#Male data plot
axes[1].scatter(df_male['Height'],df_male['Weight'],color = 'green', alpha = 0.5)
axes[1].set_title(f'Male Correlation:{corr_male.loc["Height","Weight"]: .2f}')
axes[1].set_xlabel('Height')
axes[1].set_ylabel('Weight')
axes[1].grid(True)


#female data plot
axes[2].scatter(df_female['Height'],df_female['Weight'],color = 'red', alpha = 0.5)
axes[2].set_title(f'Female Correlation:{corr_female.loc["Height","Weight"]: .2f}')
axes[2].set_xlabel('Height')
axes[2].set_ylabel('Weight')
axes[2].grid(True)

plt.tight_layout()
plt.show()