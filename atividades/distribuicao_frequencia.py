import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dados = np.array([160, 165, 167, 164, 160, 166, 160, 161, 150, 152, 173, 160, 155,
                  164, 168, 162, 161, 168, 163, 156, 155, 169, 151, 170, 164,
                  155, 152, 163, 160, 155, 157, 156, 158, 158, 161, 154, 161, 156, 172, 153])

dados = np.sort(dados)

print(dados)

minimo = dados.min()

maximo = dados.max()

unicos = np.unique(dados,return_counts=True)

plt.bar(dados,dados)

n = len(dados)

i = round(1+3.3*np.log10(n)) #numero de classes

import math
h = math.ceil((maximo-minimo)/i) #intervalo de classes

intervalos = np.arange(minimo,maximo+2,h)

intervalo1 = 0
intervalo2 = 0
intervalo3 = 0
intervalo4 = 0
intervalo5 = 0
intervalo6 = 0

for i in range(n):
    if dados[i] >= intervalos[0] and dados[i] < intervalos[1]:
        intervalo1 += 1
    elif dados[i] >= intervalos[1] and dados[i] < intervalos[2]:
        intervalo2 += 1
    elif dados[i] >= intervalos[2] and dados[i] < intervalos[3]:
        intervalo3 += 1
    elif dados[i] >= intervalos[3] and dados[i] < intervalos[4]:
        intervalo4 += 1
    elif dados[i] >= intervalos[4] and dados[i] < intervalos[5]:
        intervalo5 += 1
    elif dados[i] >= intervalos[5] and dados[i] < intervalos[6]:
        intervalo6 += 1
 
 
list_intervalo = [intervalo1,intervalo2,intervalo3,intervalo4,intervalo5,intervalo6]

lista_classes = []
for i in range(len(list_intervalo)):
    lista_classes.append(str(intervalos[i])+'--'+ str(intervalos[i+1]))
    
plt.bar(lista_classes,list_intervalo)
plt.title('Histograma')
plt.xlabel('intervalos')
plt.ylabel('valores')