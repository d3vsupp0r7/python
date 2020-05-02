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
print('** Dataset info **')
print(df.describe())
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
R1: Test statistic (TStatistic index). We can use it to compare it to determinate if we have 
    significatnt proof of stationarity
    -1.7369847452352438, 
R2: Represent the "P value" associated to TStatistic index.
    0.4121645696770621, 
R3: 
    18, 
R4: 
    5002, 
R5: Represent the 1,5,10 % of criticals values of Dickey-Fuller Table
    {'1%': -3.431658008603046, 
    '5%': -2.862117998412982,   
    '10%': -2.567077669247375}, 
R6: 39904.880607487445)

In this case:
*) TStatistics index  (R1) > of 1/5/10 % critical values
*) The "P value" (R2) represent the 41 % of no rejecting the null
*) The R3 represents the number of lags used into regression used for
    determinating the "T statistics". This means thet are some 
    autocorrelaton after x periods (18 in this case)
*) R4: Number of observations used into analisys. This number depends on
    numvber of lags used into regression.
*) R6: Represent the estimation of maximized information criteria
    provided. Lower values mean that is easier to make prediction for the future.
    
'''
print("** Example of using Dickye-Fuller on Withe Noise Time series for result comparison ***")
# -- Withe Noise processing --
#1. Generate a withe noise series
'''
wn = np.random.normal() 
'''
wn = np.random.normal(loc = df.market_value.mean(), scale = df.market_value.std(), size = len(df))
#
df['wn'] = wn
print(df.describe())
print('** Dickey-Fuller Test - On WhiteNoise **')
print(sts.adfuller(df.wn))
'''
Result of Dickye-Fuller on WhiteNoise dataset
(-28.497108423820595, 
    0.0, 
    5, 
    5015, 
    {'1%': -3.431654616214729, 
        '5%': -2.862116499672828, 
        '10%': -2.567076871409699}, 
    70900.23330187848)

**
R1: -28.497108423820595, 
R2: 0.0, 
R3: 5, 
R4: 5015, 
R5: 
    {'1%': -3.431654616214729, 
        '5%': -2.862116499672828, 
        '10%': -2.567076871409699}, 
R6: 70900.23330187848
'''
print('** Dickey-Fuller Test - On RandomWalk **')
print('*** RandomWalk processing ***')
ds_filename_rw = "../time-series_datasets/market_index_2018_RandWalk_example.csv"
#Define the complete dataframe
pd_dataset_compl_rw = pd.read_csv(ds_filename_rw)
# working dataset: pd_dataset
pd_dataset_rw = pd.read_csv(ds_filename_rw)

# Working with dataset
pd_dataset_rw.date = pd.to_datetime(pd_dataset_rw.date, dayfirst = True)
pd_dataset_rw.set_index("date", inplace=True)
pd_dataset_rw=pd_dataset_rw.asfreq('b')
print(pd_dataset_rw.describe())
df['rw']=pd_dataset_rw.price
print(df.head())
print(sts.adfuller(df.rw))

'''
Result of Dickye-Fuller on Random Walk dataset

(-1.328607392768972,
 0.6159849181617384, 
 24, 
 4996, 
 {'1%': -3.4316595802782865, 
    '5%': -2.8621186927706463, 
    '10%': -2.567078038881065}, 
    46299.333497595144)
**
R1: -1.328607392768972,
R2: 0.6159849181617384, 
R3: 24, 
R4: 4996, 
R5: {'1%': -3.4316595802782865, 
    '5%': -2.8621186927706463, 
    '10%': -2.567078038881065}, 
R6: 46299.333497595144
    
'''