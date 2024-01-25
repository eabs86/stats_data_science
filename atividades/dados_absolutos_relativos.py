import pandas as pd

dados = {'emprego':['Administrador_banco_dados','Programado','Arquiteto_redes'],
         'nova_jersey':[97350,82080,112840],
         'florida':[77140,71540,62310]}

dataset = pd.DataFrame(data=dados)

dataset['nova_jersey'].sum()

dataset['florida'].sum()

dataset['%_nova_jersey'] = dataset['nova_jersey'] / dataset['nova_jersey'].sum() * 100

dataset['%_florida'] = dataset['florida'] / dataset['florida'].sum() * 100
