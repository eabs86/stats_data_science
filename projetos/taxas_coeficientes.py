import pandas as pd

dados = {'ano_graduacao':['1°','2°','3°','4°'],
         'matriculas_marco':[70,50,47,23],
         'matriculas_novembro':[65,48,40,22]}

dataset = pd.DataFrame(data=dados)

dataset['taxa_evasao']=((dataset['matriculas_marco']-dataset['matriculas_novembro'])/dataset['matriculas_marco'])*100