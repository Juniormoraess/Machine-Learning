import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

dataset = pd.read_csv('Credit.csv')
dt = dataset.iloc[:, [1, 4, 7]].values

## Padronizando dt e alocando em x:
sc = StandardScaler()
x = sc.fit_transform(dt)

## Normalizando dt e alocando em y:
mms = MinMaxScaler()
y = mms.fit_transform(dt)