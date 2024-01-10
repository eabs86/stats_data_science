"""
- Define-se os grupos de amostras
- Não se define aleatoriamente. Define-se de acordo com
o tamanho da base, a partir do seu início.
- A seleção dos grupos é feita de maneira aleatoria


"""


import pandas as pd
import random
import numpy as np

RANDOM_SEED = 10

random.seed(RANDOM_SEED)
dataset = pd.read_csv('./assets/dados/census.csv')

tam_populacao = len(dataset)
QUANTIDADE_GRUPOS = 100
TAMANHO_GRUPO = tam_populacao// QUANTIDADE_GRUPOS

grupos = []
id_grupos = 0
contagem = 0

for _ in dataset.iterrows():
    grupos.append(id_grupos)
    contagem +=1
    if contagem > TAMANHO_GRUPO:
        id_grupos += 1
        contagem = 0

print(grupos)
np.unique(grupos,return_counts=True) #para verificar a quantidade
# detalhe que o último grupo tem menos ocorrências porque a divisão
# não é inteira.

dataset['grupo'] = grupos

grupo_selecionado = random.randint(0,QUANTIDADE_GRUPOS-1)

df_agrupamento = dataset[dataset['grupo'] == grupo_selecionado]


# criando uma função

def amostragem_agrupamento(dataset,numero_grupos,random_seed):
    
    random.seed(random_seed)
    tam_populacao = len(dataset)
    TAMANHO_GRUPO = tam_populacao// numero_grupos

    grupos = []
    id_grupos = 0
    contagem = 0

    for _ in dataset.iterrows():
        grupos.append(id_grupos)
        contagem +=1
        if contagem > TAMANHO_GRUPO:
            id_grupos += 1
            contagem = 0
        
    dataset['grupo'] = grupos

    grupo_selecionado = random.randint(0,QUANTIDADE_GRUPOS-1)

    df_agrupamento = dataset[dataset['grupo'] == grupo_selecionado]
    
    return df_agrupamento

df_agrupamento_funcao = amostragem_agrupamento(dataset,QUANTIDADE_GRUPOS)