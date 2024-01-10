""" 
Usando Naive Bays para trabalhar com dados desbalanceados

"""

import pandas as pd
import random 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


RANDOM_SEED = 1

random.seed(RANDOM_SEED)

dataset = pd.read_csv('../assets//dados//credit_data.csv')

dataset.dropna(inplace=True) #removendo os NaN
sns.set_palette(['orange','blue'])
sns.countplot(x='c#default', data=dataset);