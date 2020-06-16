import pandas as pd
from apyori import apriori

dados = pd.read_csv('transacoes.txt', header = None)

transacoes = []
for i in range(0, len(dados)):
    transacoes.append([str(dados.values[i, j]) for j in range(0, 3)])


regras = apriori(transacoes, min_support = 0.5, 
                 min_confidence = 0.5)

result = list(regras)
result2 = [list(x) for x in result]
result3 = []
for j in range(0, len(result)):
    result3.append([list(x) for x in result2[j][2]])

result3
