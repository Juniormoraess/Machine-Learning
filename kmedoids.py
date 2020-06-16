import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer


iris = datasets.load_iris()

## Executando o modelo com o dataset iris
cluster = kmedoids(iris.data[:, 0:2], [3, 12, 20])
cluster.get_medoids()
cluster.process()

## Alocando os indices das instâncias e calculando os centroids
previsoes = cluster.get_clusters()
medoids = cluster.get_medoids()

## Plotando o gráfico
v = cluster_visualizer()
v.append_clusters(previsoes, iris.data[:, 0:2])
v.append_cluster(medoids, iris.data[:, 0:2], marker = '*', 
                 markersize = 15)
v.show()

## Transformando os indices das instâncias em classificação do dataset
listPrevisoes = []
listReal = []
for i in range(0, len(previsoes)):
    for j in range(0, len(previsoes[i])):
        listPrevisoes.append(i)
        listReal.append(iris.target[previsoes[i][j]])

listPrevisoes = np.asarray(listPrevisoes)
listReal = np.asarray(listReal)

matrizConfusao = confusion_matrix(listReal, listPrevisoes)