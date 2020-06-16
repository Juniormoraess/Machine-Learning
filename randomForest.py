import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

credito = pd.read_csv('Credit.csv')
previsores = credito.iloc[:, 0:20].values
classe = credito.iloc[:, 20].values

labelencoder = LabelEncoder()

def labelEncoder(vetor):
    for i in range(0, 20):
        if type(vetor[0][i]) != type(0):
            vetor[:, i] = labelencoder.fit_transform(vetor[:, i])
        
labelEncoder(previsores)

X_treino, X_teste, y_treino, y_teste = train_test_split(previsores, 
                                                        classe,
                                                        test_size = 0.3,
                                                        random_state = 0)

floresta = RandomForestClassifier(n_estimators = 100)
floresta.fit(X_treino, y_treino)

previsoes = floresta.predict(X_teste)
confusao = confusion_matrix(y_teste, previsoes)
taxa_acerto = accuracy_score(y_teste, previsoes)

## Caso queira visualizar as arvores de decisao
## Pode-se utilizar o graphviz também
floresta.estimators_
floresta.estimators_[0]