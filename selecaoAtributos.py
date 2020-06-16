import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

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

svm = SVC()
svm.fit(X_treino, y_treino)

previsoes = svm.predict(X_teste)
taxa_acerto = accuracy_score(y_teste, previsoes)

forest = ExtraTreesClassifier()
forest.fit(X_treino, y_treino)
importancias = forest.feature_importances_

X_treino2 = X_treino[:, [0, 1, 2, 3, 4]]
X_teste2 = X_teste[:, [0, 1, 2, 3, 4]]

svm2 = SVC()
svm2.fit(X_treino2, y_treino)
previsoes2 = svm2.predict(X_teste2)


taxa_acerto2 = accuracy_score(y_teste, previsoes2)