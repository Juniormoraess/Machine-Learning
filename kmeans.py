import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

## Alocando as classificações e a quantidade correspondente a cada classificação
iris = datasets.load_iris()
unicos, quanty = np.unique(iris.target, return_counts = True)

## Chamando o método e executando ele com o dataset iris
cluster = KMeans(n_clusters = 3)
cluster.fit(iris.data)

## Calculando os centroids e a classificação resultantes das previsoes
centroides = cluster.cluster_centers_
previsoes = cluster.labels_

## Alocando as classificações e a quantidade correspondente a cada classificação da previsão
unicos2, quanty2 = np.unique(previsoes, return_counts = True)

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