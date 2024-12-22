from operator import index

import pandas as pd
df = pd.read_csv("tips.csv")
print("--------------------------\n",df.info())

print("===========================\n",df.shape)
print("++++++++++++++++++++++++++\n",df)

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",df.index)


print("??????????????????????\n",df.columns)

if df.shape[0] > 0 and  df.shape[1] > 0:
    value = df.iat[0,0]
    print("@@@@@@@@@@@@@@@@@@@@@\nValue at [0,0] is ",value)
else:
    print("DataFrame is empty or  has no columns")


print("--------------------\n")

print( df.shape[0]) #no. of rows
print(df.shape[1]) #no. of columns

df1 = df.iat[0,0]
print("::::::::::::::::::\n",df1)
print(">>>>>>>>>>>>>>>>>>>>>\n",df.iloc[0,0])

