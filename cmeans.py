import numpy as np
import skfuzzy as sk
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()

## Chamando o método e executando ele com o dataset iris
r = sk.cmeans(data = iris.data.T, c = 3, m = 2, error = 0.005, 
              maxiter = 1000, init = None)

## Recebendo os valores em porcentagem e pegando a classificação de cada valor
previsoesPorcent = r[1]
previsoes = previsoesPorcent.argmax(axis = 0)
matrizConfusao = confusion_matrix(iris.target, previsoes)

## Plotando o gráfico
plt.scatter(iris.data[previsoes == 0, 0], 
            iris.data[previsoes == 0, 1],
            c = 'green', label = 'Setosa')
plt.scatter(iris.data[previsoes == 1, 0], 
            iris.data[previsoes == 1, 1],
            c = 'red', label = 'Versicolor')
plt.scatter(iris.data[previsoes == 2, 0], 
            iris.data[previsoes == 2, 1],
            c = 'blue', label = 'Virginica')
plt.legend()