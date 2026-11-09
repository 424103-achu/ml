import numpy as np 
import pandas as pd
import matplotlib.pyplot as mp



a = np.random.random((5,5))
print(a)
a[a<0.5] = 1
print(a)

df = pd.read_csv('data.csv', index_col=0)
print(df.info())
print(df.describe())

t = len(df) * 0.5
df = df.dropna(thresh=t,axis=1)

print(df)


df1 = pd.read_csv('data2.csv')
df1.plot(subplots=True, layout=(2,2))
mp.show()