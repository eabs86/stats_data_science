import pandas as pd
import random
import numpy as np

RANDOM_SEED = 10

random.seed(RANDOM_SEED)
dataset = pd.read_csv('./assets/dados/census.csv')

tam_populacao = len(dataset)

AMOSTRA = 100

razao = (tam_populacao // AMOSTRA)

primeira_amostra = random.randint(0,razao)

amostras = []
amostras.append(dataset.iloc[primeira_amostra])

#Formas de selecionar:
# 1 - Por Esrutura de Repetição
for count in range(1,AMOSTRA):
    print(primeira_amostra+razao*count)
    amostras.append(dataset.iloc[primeira_amostra+razao*count])

amostras = pd.DataFrame(amostras)
    
# 2 - Por Numpy

indices = np.arange(primeira_amostra, tam_populacao, step=razao)

amostras_nummpy = dataset.iloc[indices]

# 3 - Criando uma função

def amostragem_sistematica(dataset, amostra,random_seed):
    random.seed(random_seed)
    tam_populacao = len(dataset)
    razao = (tam_populacao // amostra)
    primeira_amostra = random.randint(0,razao)
    indices = np.arange(primeira_amostra, tam_populacao, step=razao)
    amostras_nummpy = dataset.iloc[indices]
    
    return amostras_nummpy

grupo_amostras = amostragem_sistematica(dataset, AMOSTRA, RANDOM_SEED)