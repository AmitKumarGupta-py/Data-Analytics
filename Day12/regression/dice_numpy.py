

#part1
import numpy as np
def roll_dice():
    return np.sum(np.random.randint(1,7,2))
print(roll_dice())

#simulation of rolling dice twice , minimum number we get is 1 max is 6
# add the results of the two simulations so possible values will be
#(1,1)...
#run multiple times to verify



#part2
#Someone approaches us saying I will give you % dollars if you get 7
#and take 1 dollar if you get a number other than 7

#how do we know what will happen?
#our own "Monte carlo Simulation " like function

def monte_carlo_simulation(runs=1000):
    results = np.zeros(2) #An array , results[0] and results[1] initialized to two zeroes
    for _ in range(runs):
        if roll_dice() == 7:
            results[0] += 1
        else:
            results[1] += 1
    return results

print(monte_carlo_simulation())
print(monte_carlo_simulation())
print(monte_carlo_simulation())


#part 3: Now do it 1000 times : take some time
results = np.zeros(1000)

for i in range(1000):
    results[i] = monte_carlo_simulation()[0]
print(results)

#Let us plot it
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(results, bins=15)
plt.show()

#Our win/loss
print(results.mean()) #General mean
print(results.mean() * 5) #what we will get as win on an average
print(results.mean() * 4.75) #
print(1000 - results.mean()) #what we will pay on average

print(results.mean()/1000)  #probability of the 'we will win ' result

#the last probability should be close to the theoreical probability of getting
#a 7 when we throw two dice 