import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from sympy.stats.rv import probability

tips_df = pd.read_csv("tips.csv")

#binary outcome
def classify_tip(row):
    total_bill = row['total_bill']
    tip = row['tip']
    if(tip/total_bill) > 0.15:
        return  1
    else:
        return 0

#apply the function to create the binary column
tips_df['tip_binary'] = tips_df.apply(classify_tip, axis = 1)

#total no.
n = len(tips_df)
#no. of success
k = tips_df['tip_binary'].sum()
# probability of success
p = k/n

trails = 100

x = np.arange(0,trails + 1)
#calculate the pmf for each number of successes
pmf_values = binom.pmf(x,trails,p)

cumulative_probability = 0

for i in x:
    cumulative_probability = binom.cdf(i,trails,p)
    print(f"Probability for {i} successes = {pmf_values[i]:.6f} ... cumulative probability = {cumulative_probability:.6f}")