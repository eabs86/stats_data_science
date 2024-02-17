import pandas as pd
import random
import numpy as np

RANDOM_SEED = 10

random.seed(RANDOM_SEED)
dataset = pd.read_csv('../assets/dados/census.csv')

dataset.head(10)

# para usar regras de associação não pode ser dados numériocos
# é preciso que os dados sejam strings


dataset['age'].max(), dataset['age'].min()

dataset['age'].plot.hist()


dataset['age'] = pd.cut(dataset['age'], bins = [0, 17, 25, 40, 60,90],
                        labels=['Faixa1', 'Faixa2', 'Faixa3', 'Faixa4', 'Faixa5'])

dataset['age'].unique()

#após transformar em faixas, vamos fazer a regra de associação

dataset_apriori = dataset[['age','workclass','education','marital-status',
                         'occupation','relationship','sex','native-country','income']]


dataset_apriori.head(10)

dataset_apriori.shape

dataset_apriori_sample = dataset_apriori.sample(n=1000)

dataset_apriori_sample.shape

transacoes = []

for i in range(dataset_apriori_sample.shape[0]):
    transacoes.append([str(dataset_apriori_sample.values[i,j]) for j in range(dataset_apriori_sample.shape[1])])
    
transacoes

from apyori import apriori

regras = apriori(transacoes,min_support = 0.3,min_confidence = 0.2,min_lift = 1)
resultados = list(regras)


resultados[12]