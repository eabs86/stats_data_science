import pandas as pd
import random 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


RANDOM_SEED = 1

random.seed(RANDOM_SEED)

dataset = pd.read_csv('..//assets//dados//csv_result-ebay_confianca_completo.csv')

dataset['blacklist']= dataset['blacklist']=='S'

sns.countplot(x=dataset['reputation'])

X = dataset.iloc[:,1:74].values

y= dataset.iloc[:,74].values

np.unique(y,return_counts=True)

from sklearn.model_selection import train_test_split

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y )

from sklearn.ensemble import RandomForestClassifier

modelo = RandomForestClassifier()
modelo.fit(X_treinamento,y_treinamento)

from sklearn.metrics import accuracy_score, confusion_matrix

acuracia = accuracy_score(y_teste,modelo.predict(X_teste))

matriz_confusao = confusion_matrix(y_teste,modelo.predict(X_teste))

print(matriz_confusao)

print(acuracia)




#subamostragem com tomeklinks

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

modelo_under = RandomForestClassifier()
modelo_under.fit(X_train_under,y_train_under)
previsoes_under = modelo_under.predict(X_test_under)
accuracy_score_under = accuracy_score(y_test_under,previsoes_under)

print(accuracy_score_under)

#sobreamostragem

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')

X_smote, y_smote = smote.fit_resample(X, y)

X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(X_smote, y_smote,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y_smote )

modelo_smote = RandomForestClassifier()
modelo_smote.fit(X_train_over,y_train_over)
previsoes_smote = modelo_smote.predict(X_test_over)
accuracy_score_smote = accuracy_score(y_test_over,previsoes_smote)
print(accuracy_score_smote)
