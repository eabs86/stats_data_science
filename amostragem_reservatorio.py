"""
Amostra de Reservatório:

É um tipo de amostragem de dados indicada para Data Stream de itens
com tamanho desconhecido, que pode ser acessado somente uma vez.
    - Não é uma base de dados estática, e sim dinâmica
    
- Algoritmo para sortear um item do stream, porém, cada item deve
possuir a mesma probabilidade de seleção.

"""


from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd
import random
import numpy as np

RANDOM_SEED = 10

random.seed(RANDOM_SEED)

dataset = pd.read_csv('./assets/dados/census.csv')
stream = []

for i in range(len(dataset)):
    stream.append(i)
    
print(stream)


def amostragem_reservatorio(dataset,amostras):
    stream = []
    for i in range(len(dataset)):
        stream.append(i)
    
    i=0
    tamanho = len(dataset)
    
    reservatorio = [0]*amostras
    for i in range(amostras):
        reservatorio[i]=stream[i]
    
    while i<tamanho:
        j = random.randrange(i+1)
        if j <amostras:
            reservatorio[j] = stream[i]
        i+=1
    return dataset.iloc[reservatorio]

df = amostragem_reservatorio(dataset,100)

df.shape