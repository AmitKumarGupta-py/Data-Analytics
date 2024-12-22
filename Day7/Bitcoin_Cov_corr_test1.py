import pandas as pd
import matplotlib.pyplot as plt

from covariance_test1 import corr_overall

df = pd.read_csv("SP500 oil gold bitcoin.csv")
print(df)


corr_gold_bitcoin = df[['Gold','BITCOIN']].corr()
corr_gold_sp500 = df[['Gold','S&P500']].corr()
corr_bitcoin_sp500 = df[['BITCOIN','S&P500']].corr()


cov_gold_bitcoin = df[['Gold','BITCOIN']].cov()
cov_gold_sp500 = df[['Gold','S&P500']].cov()
cov_bitcoin_sp500 = df[['BITCOIN','S&P500']].cov()

print("Correlation of Gold and Bitcoin: ",corr_gold_bitcoin)
print("Correlation of Gold and S&P500: ",corr_gold_sp500)
print("Correlation of S&p500and Bitcoin: ",corr_bitcoin_sp500)
print("Covariance of Gold and Bitcoin: ",cov_gold_bitcoin)
print("Covariance of Gold and S&P500: ",cov_gold_sp500)
print("Covariance of S&P500 and Bitcoin: ",cov_bitcoin_sp500)

fig, axes = plt.subplots(1,3, figsize = (18,6))

#Overall DATA plot
xmin,xmax = axes[0].get_xlim()
ymin,ymax = axes[0].get_ylim()

text_x = xmin + 0.7 * (xmax - xmin)
text_y = ymin + 0.3 * (ymax - ymin)
plt.text(text_x,text_y,f"Correlation:{corr_gold_bitcoin.loc['BITCOIN','Gold']: .2f}",ha = 'center', va = 'center', fontsize = 12)

axes[0].scatter(df['Gold'],df['BITCOIN'],color = 'blue', alpha = 0.5)
axes[0].set_title(f'Correlation of Gold and Bitcoin:{corr_gold_bitcoin.loc["Gold","BITCOIN"]: .2f}')
axes[0].set_xlabel('Gold')
axes[0].set_ylabel('BITCOIN')
axes[0].grid(True)

xmin,xmax = axes[1].get_xlim()
ymin,ymax = axes[1].get_ylim()
text_x = xmin + 0.7 * (xmax - xmin)
text_y = ymin + 0.3 * (ymax - ymin)

plt.text(text_x,text_y,f"Correlation:{corr_gold_sp500.loc['S&P500','Gold']: .2f}",ha = 'center', va = 'center', fontsize = 12)

axes[1].scatter(df['Gold'],df['S&P500'],color = 'blue', alpha = 0.5) #alpha =0.5 means semi transparent
axes[1].set_title(f'Correlation of Gold and S&P500:{corr_gold_sp500.loc["Gold","S&P500"]: .2f}')
axes[1].set_xlabel('Gold')
axes[1].set_ylabel('S&P500')
axes[1].grid(True)

xmin,xmax = axes[2].get_xlim()
ymin,ymax = axes[2].get_ylim()
text_x = xmin + 0.7 * (xmax - xmin)
text_y = ymin + 0.3 * (ymax - ymin)
axes[2].scatter(df['S&P500'],df['BITCOIN'],color = 'blue', alpha = 0.5)
axes[2].set_title(f'Correlation of S&P500 and Bitcoin:{corr_bitcoin_sp500.loc["S&P500","BITCOIN"]: .2f}')
axes[2].set_xlabel('S&P500')
axes[2].set_ylabel('BITCOIN')
axes[2].grid(True)

plt.tight_layout()

# #xmin, xmax = plt.xlim()
# ymin, ymax = plt.ylim()
#
# text_x = xmin + 0.7 * (xmax - xmin)
# text_y = ymin + 0.3 * (ymax - ymin)

#plt.text(text_x,text_y,f"Correlation:{corr_gold_bitcoin.loc['BITCOIN','Gold']: .2f}",ha = 'center', va = 'center', fontsize = 12)


plt.show()

