import pandas as pd
import random
import numpy as np

dataset = pd.read_csv('./assets/dados/census.csv')

df_amostra_aleatoria_simples = dataset.sample(n=100)

def amostragem_aleatoria_simples(dataset, nr_amostras,random_state):
    return dataset.sample(n=nr_amostras,random_state=random_state)

df_amostra_aleatoria_simples = amostragem_aleatoria_simples(dataset,100,40)