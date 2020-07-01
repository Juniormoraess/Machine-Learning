import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


base = pd.read_csv('questao2_creditcard.csv', sep = ';')
titulos = base.iloc[0, 0:24]
base = base.drop([0])
previsores = base.iloc[:, 0:23].values
classe = base.iloc[:, 23].values



X_treino, X_teste, y_treino, y_teste = train_test_split(previsores, 
                                                        classe,
                                                        test_size = 0.3, 
                                                        random_state = 0)

forest = ExtraTreesClassifier()
forest.fit(X_treino, y_treino)
importancias = forest.feature_importances_

X_treino2 = X_treino[:, [0, 4, 5]] 
X_teste2 = X_teste[:, [0, 4, 5]]

base2 = pd.read_csv('questao22_creditcard_clientes.csv', sep = ';')
base2 = base2.drop([0])
novosClientes = base2.iloc[:, [0, 4, 5]].values
base3 = np.asarray(base2)

def svm(x):
    svm = SVC()
    svm.fit(X_treino2, y_treino)
    previsoes = svm.predict(x)
    return previsoes
    
def naiveBayes(x):
    naiveBayes = GaussianNB()
    naiveBayes.fit(X_treino2, y_treino)
    previsoes = naiveBayes.predict(x)
    return previsoes

def randomForest(x):
    floresta = RandomForestClassifier(n_estimators = 1000)
    floresta.fit(X_treino2, y_treino)
    previsoes = floresta.predict(x)
    return previsoes
    

result = randomForest(novosClientes)


with open('previsoes.csv', 'w') as arquivo_csv:
    colunas = ['LIMIT_BAL', 'AGE', 'PAY_0', 'default_payment']
    escrever = csv.DictWriter(arquivo_csv, fieldnames = colunas, 
                              delimiter = ';', lineterminator='\n')
    
    escrever.writeheader()
    for j in range(0, len(base3)):
        escrever.writerow({'LIMIT_BAL': base3[j][0],
                           'AGE': base3[j][4],
                           'PAY_0': base3[j][5],
                           'default_payment': result[j]})


clientesCobrados = pd.read_csv('previsoes.csv', sep = ';')
clientesCobrados.sort_values(by = ['default_payment','LIMIT_BAL', 'PAY_0'],
                             ascending = [True, False, False], 
                             inplace = True)
clientesCobrados = clientesCobrados.iloc[0:272, :]
clientesCobrados.to_csv('clientesCobrados.csv', sep = ';')

