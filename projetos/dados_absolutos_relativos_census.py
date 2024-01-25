import pandas as pd
import random
import numpy as np

RANDOM_SEED = 10

random.seed(RANDOM_SEED)
dataset = pd.read_csv('../assets/dados/census.csv')

dataset_2 = dataset[['income','education']]

dataset_3 = dataset_2.groupby(['education','income'])['education'].count()

dataset_3[' Bachelors', ' <=50K'], dataset_3[' Bachelors', ' >50K']

3134 + 2221

percentual_acima50k = 2221 / (3134 + 2221) 

print(f'Percentual de pessoas que ganha acima de 50K e tem ensino superior: {percentual_acima50k*100} %')

percentual_abaixo50k = 3134 / (3134 + 2221)

print(f'Percentual de pessoas que ganha abaixo de 50K e tem ensino superior: {percentual_abaixo50k*100} %')