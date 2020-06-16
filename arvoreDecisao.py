import pandas as pd
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

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

arvore = DecisionTreeClassifier()
arvore.fit(X_treino, y_treino)

export_graphviz(arvore, out_file = 'tree.dot')

previsoes = arvore.predict(X_teste)
confusao = confusion_matrix(y_teste, previsoes)
taxa_acerto = accuracy_score(y_teste, previsoes)