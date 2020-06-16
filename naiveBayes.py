import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from yellowbrick.classifier import ConfusionMatrix

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


naive_bayes = GaussianNB()
naive_bayes.fit(X_treino, y_treino)

previsoes = naive_bayes.predict(X_teste)
confusao = confusion_matrix(y_teste, previsoes)
taxa_acerto = accuracy_score(y_teste, previsoes)

v = ConfusionMatrix(GaussianNB())
v.fit(X_treino, y_treino)
v.score(X_teste, y_teste)
v.poof()

novo_credito = pd.read_csv('NovoCredit.csv')
novo_credito = novo_credito.iloc[:, 0:20].values
labelEncoder(novo_credito)

nova_previsao = naive_bayes.predict(novo_credito)

print()
print('Seu novo cliente e: {} pagador'.format(nova_previsao[0]))
print()