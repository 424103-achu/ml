import pandas as pd
dfwhite = pd.read_csv("winequality-white.csv",sep=';')
dfred = pd.read_csv("winequality-red.csv")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(len(dfwhite),len(dfred)," this is the number of values from initail data")

#first remove duplicate data
dfwhitenew = dfwhite.drop_duplicates()
dfrednew = dfred.drop_duplicates()

print(len(dfwhitenew), len(dfrednew) ," this is the number of values from after deleting duplicate data")

#then drop N/A
dfwhite1 = dfwhitenew.dropna()
dfred1 = dfrednew.dropna()

print(len(dfwhite1),len(dfred1)," this is the number of rows after dropping NA values")

#now we split into train and test
whitetrain = dfwhite1.sample(frac=0.8,random_state=6969)
whitetest = dfwhite1.drop(whitetrain.index)

redtrain = dfred1.sample(frac=0.8,random_state=6969)
redtest = dfred1.drop(redtrain.index)

print(len(whitetrain),len(whitetest))
print(len(redtrain),len(redtest))


X_train = whitetrain.drop(columns=['quality'])
Y_train = whitetrain['quality']
print(X_train)
print(Y_train)
X_test=whitetest.drop(columns=["quality"])
Y_test=whitetest["quality"]
scaler = StandardScaler()
x_new = scaler.fit_transform(X_train)
x_new_test=scaler.transform(X_test)
print(x_new)
