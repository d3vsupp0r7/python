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
#
del df_comp['spx']
del df_comp['dax']
del df_comp['ftse']
del df_comp['nikkei']
size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]
# -- Withe Noise processing --
#1. Generate a withe noise series
'''
wn = np.random.normal() 
'''
wn = np.random.normal(loc = df.market_value.mean(), scale = df.market_value.std(), size = len(df))
#
df['wn'] = wn
print(df.describe())
#
df.wn.plot()
plt.title("White Noise - Time Series spx", size=24)
plt.show()
#
df.wn.plot(figsize=(20,5))
plt.title("White Noise - Time Series spx - figsize", size=24)
plt.show()
#
df.market_value.plot(figsize=(20,5))
plt.title("Original S&P prices", size=24)
plt.show()
#
df.market_value.plot(figsize=(20,5))
plt.title("Original S&P prices - ylim", size=24)
plt.ylim(0,2300)
plt.show()
#
#
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
# Plot the random walk
df.rw.plot(figsize=(20,5))
plt.title("Random Walk - Time Series spx - figsize", size=24)
plt.show()
#Plot togheter
df.rw.plot(figsize=(20,5))
df.market_value.plot()
plt.title("Random Walk vs S&P - figsize", size=24)
plt.show()