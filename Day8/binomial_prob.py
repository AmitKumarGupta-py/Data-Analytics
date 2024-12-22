from scipy.stats import binom
from sympy.stats.rv import probability

n,k,p = 5,3,0.66 #n: total no. of students
#k: number of students who like python
#p:probability that a student like python

probability1 = binom.pmf(k,n,p)

print(f"The probability that exactly {k} out of {n} students like Python is: {probability1:.4f}")

#probability : given that 2 out of 7 students prefer online
# #learning over in person classes when, general, 55% of students prefer
#online learning
n=7
k=2
p = 0.55

probability2 = binom.pmf(k,n,p)

print(f"\n\nThe probability that exactly {k} out of {n} students like online learning over in person classes is: {probability2:.4f}")

probability3 = binom.pmf(k,n,1-p)

print(f"\n\nThe probability that exactly {k} out of {n} students not like online learning over in person classes is: {probability3:.4f}")