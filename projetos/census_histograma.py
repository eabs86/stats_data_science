import pandas as pd
import random
import numpy as np

RANDOM_SEED = 10

random.seed(RANDOM_SEED)
dataset = pd.read_csv('../assets/dados/census.csv')

dataset.head()

dataset['age'].max(), dataset['age'].min()

dataset['age'].plot.hist()


dataset['age'] = pd.cut(dataset['age'], bins = [0, 17, 25, 40, 60,90],
                        labels=['Faixa1', 'Faixa2', 'Faixa3', 'Faixa4', 'Faixa5'])

dataset['age'].unique()

