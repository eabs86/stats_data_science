"""
Na amostra estratificada, as amostras são escolhidas com base
na proporção de cada classe contida na população/dataset.

Depois de separada as proporções (quantidades que se deve selecionar de cada classe na população),
deve-se utilizar outra técnica para seleção de amostras.

"""

from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd
import random
import numpy as np

RANDOM_SEED = 10

random.seed(RANDOM_SEED)
dataset = pd.read_csv('./assets/dados/census.csv')

dataset['income'].value_counts()

# income
#  <=50K    24720  => Representa 75,92% da população
#  >50K      7841  => Representa 24,08% da população

# extraindo uma amostra estratificada

split = StratifiedShuffleSplit(test_size=0.1, random_state=RANDOM_SEED)

for x,y in split.split(dataset,dataset['income']):
    df_x = dataset.iloc[x]
    df_y = dataset.iloc[y]

df_x['income'].value_counts()/len(dataset)
df_y['income'].value_counts()/len(dataset)