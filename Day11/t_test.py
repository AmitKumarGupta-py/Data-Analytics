import pandas as pd
from mpmath import degree
from scipy import stats
from sipbuild.generator.parser.rules import p_value

data = pd.read_csv('Mall_Customers.csv')
age_data = data['Age']

#Hypothesized population average age

pop_avg_age = 40
#perform one_sample t-test

t_statistic, p_value = stats.ttest_1samp(age_data,pop_avg_age)
degrees_of_freedom = len(age_data) - 1

#print the results

print("T-statistic: ",t_statistic)
print("P-value:",p_value)
print("Degrees of freedom:",degrees_of_freedom)

#interpretation
alpha = 0.025
if p_value < alpha:
    print("the null hypothesis (mean age = 40) is rejected.")
else:
    print("The null hypothesis (mean age =40) cannot be rejected")

critical_value = stats.t.ppf(1-alpha,degrees_of_freedom) #ppf cummulative probability (prabobility point function)

print("Critical value:",critical_value)

#two tailed t test (because we are just checking if sample mean = population mean)

#so we need to taje absolute value of t statistic

if(abs(t_statistic) < critical_value):
    print("The null hypothesis (mean age = 40) cannot be rejected")
else:
    print("the null hypothesis (mean age = 40 ) is rejected.")


print("----------------------independent two sample T- test-----------------------------------")
##### independent two sample T- test

#Separate data by genre

male_spending = data[data['Genre'] == 'Male']['Spending Score (1-100)']
female_spending = data[data['Genre'] == 'Female']['Spending Score (1-100)']

#perform two sample t test
t_statistic, p_value = stats.ttest_ind(male_spending,female_spending)

n_male = len(male_spending)
n_female = len(female_spending)

degrees_of_freedom = n_male + n_female -2

alpha = 0.025

#Find the critical value

critical_value = stats.t.ppf(1 - alpha, degrees_of_freedom)

#print the results

print("T-statistics :",t_statistic)
print("p-value:",p_value)
print("Critical value:",critical_value)

if p_value < alpha:
    print("There is a significant difference is spending score b/w male and female groups")
else:
    print("There is no significant difference is spending score b/w male and female groups ")

if abs(t_statistic) > critical_value:
    print("There is a significant difference is spending score b/w male and female groups")
else:
    print("there is no significant   difference is spending score b/w male and female groups")