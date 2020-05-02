import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose

#
ds_filename = "../time-series_datasets/market_index_2018.csv"

#Define the complete dataframe
pd_dataset_compl = pd.read_csv(ds_filename)
# working dataset: pd_dataset
pd_dataset = pd.read_csv(ds_filename)

#
df_comp=pd_dataset.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
df_comp.set_index("date", inplace=True)
df_comp=df_comp.asfreq('b')
df_comp=df_comp.fillna(method='ffill')
#
df_comp['market_value']=df_comp.spx

#Dickey-Fuller test
print('** Stationality analisys **')
print('** Dickey-Fuller Test **')
print(sts.adfuller(df_comp.market_value))

'''
** Stationality analisys **
** Dickey-Fuller Test **
(1.1194986381728387, 
0.9953796476080692, 
10, 
6266, 
{'1%': -3.4313940441948807, '5%': -2.8620013751563964, '10%': -2.567015587023998}, 
50258.20986775002)

'''

del df_comp['spx']
del df_comp['dax']
del df_comp['ftse']
del df_comp['nikkei']
size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]

#Dickey-Fuller test
print('** Stationality analisys - 2  **')
print('** Dickey-Fuller Test - 2 **')
print(sts.adfuller(df.market_value))

'''
** Stationality analisys - 2  **
** Dickey-Fuller Test - 2 **
(-1.7369847452352438, 
0.4121645696770621, 
18, 
5002, 
{'1%': -3.431658008603046, 
'5%': -2.862117998412982, 
'10%': -2.567077669247375}, 
39904.880607487445)

Data explanation for Dickey-Fuller test:
R1: -1.7369847452352438, 
R2: 0.4121645696770621, 
R3: 18, 
R4: 5002, 
R5:
    {'1%': -3.431658008603046, 
    '5%': -2.862117998412982,   
    '10%': -2.567077669247375}, 
R6: 39904.880607487445)

'''