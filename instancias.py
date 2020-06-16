from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from scipy import stats

iris = datasets.load_iris()
stats.describe(iris.data)

previsores = iris.data
classe = iris.target

X_treino, X_teste, y_treino, y_teste = train_test_split(previsores,
                                                        classe,
                                                        test_size = 0.3,
                                                        random_state = 0)


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_treino, y_treino)

previsoes = knn.predict(X_teste)
confusao = confusion_matrix(y_teste, previsoes)
taxa_acerto = accuracy_score(y_teste, previsoes)