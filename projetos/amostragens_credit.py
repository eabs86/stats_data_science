import pandas as pd
import random
import numpy as np

RANDOM_SEED = 1

random.seed(RANDOM_SEED)

dataset = pd.read_csv('../assets//dados//credit_data.csv')

dataset.shape

dataset.head()

def amostragem_aleatoria_simples(dataset,quantidade):
    return dataset.sample(n=quantidade)

df_amostra_aleatoria_simples = amostragem_aleatoria_simples(dataset,1000)

df_amostra_aleatoria_simples.shape

def amostragem_sistematica(dataset,quantidade,random_seed):

    intervalo = len(dataset)//quantidade
    random.seed(random_seed)
    inicio = random.randint(0,intervalo)
    indices = np.arange(inicio,len(dataset),step = intervalo)
    amostras_sistematicas = dataset.iloc[indices]
    return amostras_sistematicas

df_amostra_sistematica = amostragem_sistematica(dataset,1000,RANDOM_SEED)

df_amostra_sistematica.shape


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

    grupo_selecionado = random.randint(0,numero_grupos-1)

    df_agrupamento = dataset[dataset['grupo'] == grupo_selecionado]
    
    return df_agrupamento

df_amostragem_grupos = amostragem_agrupamento(dataset,2,RANDOM_SEED)
df_amostragem_grupos.shape, df_amostragem_grupos['grupo'].value_counts()

from sklearn.model_selection import StratifiedShuffleSplit

def amostragem_estratificada(dataset,proporcao,random_seed,campo):
    split = StratifiedShuffleSplit(test_size=proporcao, random_state=random_seed)
    for _,y in split.split(dataset,dataset[campo]):
        df_y = dataset.iloc[y]
    return df_y

df_amostragem_estratificada = amostragem_estratificada(dataset,0.5,RANDOM_SEED,'c#default')

df_amostragem_estratificada.shape 

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

df_amostra_reservatorio = amostragem_reservatorio(dataset,1000)

df_amostra_reservatorio.shape

dataset['age'].mean(), dataset['income'].mean(),dataset['loan'].mean()
df_amostra_aleatoria_simples['age'].mean(),df_amostra_aleatoria_simples['income'].mean(),df_amostra_aleatoria_simples['loan'].mean()
df_amostra_sistematica['age'].mean(),df_amostra_sistematica['income'].mean(),df_amostra_sistematica['loan'].mean()
df_amostragem_grupos['age'].mean(),df_amostragem_grupos['income'].mean(),df_amostragem_grupos['loan'].mean()
df_amostragem_estratificada['age'].mean(),df_amostragem_estratificada['income'].mean(),df_amostragem_estratificada['loan'].mean()
df_amostra_reservatorio['age'].mean(),df_amostra_reservatorio['income'].mean(),df_amostra_reservatorio['loan'].mean()