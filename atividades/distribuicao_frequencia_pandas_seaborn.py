import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dados = np.array([160, 165, 167, 164, 160, 166, 160, 161, 150, 152, 173, 160, 155,
                  164, 168, 162, 161, 168, 163, 156, 155, 169, 151, 170, 164,
                  155, 152, 163, 160, 155, 157, 156, 158, 158, 161, 154, 161, 156, 172, 153])



dataset = pd.DataFrame(dados, columns=['dados'])

dataset.plot.hist()
sns.histplot(dataset, kde = True)