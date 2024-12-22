import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
from scipy.stats import skew
sample_size = 100000
ratings = np.random.randint(1,6,sample_size)

right_skew_data  = np.concatenate((ratings,np.random.randint(1,2,sample_size // 2)))

left_skew_data = np.concatenate((ratings,np.random.randint(4,6,sample_size // 2)))

#func to calculate mean median and skewness
def calculate_state(data):
    mean1 = np.mean(data)
    median1 = np.median(data)
    skewness = skew(data)
    return mean1, median1,skewness

#function to plot histogram with mean and median lines

def plot_histogram(ax, data, title):
    ax.hist(data, bins = np.arange(1,7) - 0.5, rwidth = 0.8, alpha = 0.7)
    ax.set_title(title)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Frequency')
    ax.set_xticks(range(1,6))
    ax.grid(axis='y', alpha = 0.5)


    #calculate mean median and skewness

    mean1, median1, skewness = calculate_state(data)
    ax.axvline(x = mean1, color = 'red', linestyle = '--',label = f'Mean:{mean1:.2f}' )
    ax.axvline(x = median1, color = 'blue', linestyle = '--',label = f'Median:{median1}')
    ax.legend()


    print(f"{title}:")
    print(f"Mean:{mean1:.2f}")
    print(f"Median: {median1}")
    print(f"Skewness: {skewness:.2f}")

fig,axs = plt.subplots(1,3, figsize = (15,5))
plot_histogram(axs[0], ratings, 'No Skew')
plot_histogram(axs[1], right_skew_data,'Right Skew')
plot_histogram(axs[2],left_skew_data,'Left Skew')
plt.tight_layout()
plt.show()
