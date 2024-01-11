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
sns.countplot(x='c#default', data=dataset,palette={'0': 'orange', '1': 'blue'})

#armazenando os previsores

X = dataset.iloc[:,1:4].values

#armazenando o target

y = dataset.iloc[:,4].values

#criando os grupos de treinamento e teste

from sklearn.model_selection import train_test_split

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size=0.20,
                                                                  random_state=RANDOM_SEED,
                                                                  stratify=y )

from sklearn.naive_bayes import GaussianNB

#criando o modelo
modelo = GaussianNB()
#treinando o modelo
modelo.fit(X_treinamento,y_treinamento)
#realizando as previsões
previsoes = modelo.predict(X_teste)

from sklearn.metrics import accuracy_score, confusion_matrix

acuracia = accuracy_score(y_teste,previsoes)
matriz_confusao = confusion_matrix(y_teste,previsoes)

print(matriz_confusao)
print(acuracia)

sns.heatmap(matriz_confusao,annot=True)


classe_0 = 337/(337+25)
print(f'Percentual de acerto para pessoas que pagam o emprestimo: {classe_0}')

classe_1 = 33/(33+6)
print(f'Percentual de acerto para pessoas que não pagam o emprestimo: {classe_1}')

labels = ['Classe 0', 'Classe 1'] 

# Imprimir a matriz de confusão
print(confusion_matrix(y_teste, previsoes))

# Acessar os rótulos diretamente da matriz de confusão
tn, fp, fn, tp = confusion_matrix(y_teste, previsoes).ravel()

print(f'Classe 0 Verdadeira Negativa (TN): {tn}')
print(f'Classe 0 Falsa Positiva (FP): {fp}')
print(f'Classe 1 Falsa Negativa (FN): {fn}')
print(f'Classe 1 Verdadeira Positiva (TP): {tp}')

#Usando subamostragem com undersampling - Tomek links

from imblearn.under_sampling import TomekLinks

tl = TomekLinks(sampling_strategy='majority')
X_under, y_under = tl.fit_resample(X, y)

ids_under = tl.sample_indices_

np.unique(y,return_counts=True)
np.unique(y_under,return_counts=True)

X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X_under, y_under,
                                                                            test_size=0.20,
                                                                            random_state=RANDOM_SEED,
                                                                            stratify=y_under )

modelo_under = GaussianNB()
modelo_under.fit(X_train_under,y_train_under)
previsoes_under = modelo_under.predict(X_test_under)

acuracia_under = accuracy_score(y_test_under,previsoes_under)

print(acuracia_under)

cm_under = confusion_matrix(y_test_under, previsoes_under)

sns.heatmap(cm_under,annot=True)

classe_0_under = 313/(313+22)
print(f'Percentual de acerto para pessoas que pagam o emprestimo (undersampling): {classe_0_under}')

classe_1_under = 35/(35+10)
print(f'Percentual de acerto para pessoas que não pagam o emprestimo(undersampling): {classe_1_under}')