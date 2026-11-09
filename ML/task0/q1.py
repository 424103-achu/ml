import pandas as pd
df = pd.read_csv("ex2.csv")
df = df.drop_duplicates()
df = df.dropna()
print(df)
df = pd.get_dummies(df,columns=['ContractType'],dtype=int)
df['Churn'] = df['Churn'].map({'Yes':1 , 'No':0})
df['Churn'] = df.pop('Churn')
print(df)

train = df.sample(frac=0.8,random_state=222)
test = df.drop(train.index)
print(len(train),len(test))








