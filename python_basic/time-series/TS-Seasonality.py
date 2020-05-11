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

del df_comp['spx']
del df_comp['dax']
del df_comp['ftse']
del df_comp['nikkei']
size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]

#Seasonality decomposition
s_dec = seasonal_decompose(df.market_value)
print(s_dec)
#Additive
s_dec_additive = seasonal_decompose(df.market_value, model="additive")
s_dec_additive.plot()
'''plt.title("Seasonality - Time Series spx", size=24)'''
plt.show()

'''
Additive plot explanation:
In this case, the observed series and trend look like the observed series.
This because the decomposition function uses the previous period values
as trend setter.

The seasonal patterns we can observe other prices as better predictors.
As example, we can determinate a better time indicator.
If prices are higher at beginning of the mont compare to the end, we can use/compare values
from 30 periods ago.

The trend part of decomposition explains the variability of the data.
 
 Seasonal part looks like a rectangular. This happens when values
 oscillating up & down and the figure size is too small.
 This means that is no concrete cyclical pattern when using naive decomposition.
 
 Residual are errors of our model estimates.
 Essentially they are the difference between true values and prediction for any periods.
 If we note the graph, the residual are very great into 2000 and 2008, the two periods of great 
 financial instability ( new millennium/house pricing bubble).
 
 Using the additive approach on this data suggest no seasonality in data.
'''
#Multiplicative approach
s_dec_multiplicative = seasonal_decompose(df.market_value, model="multiplicative")
s_dec_multiplicative.plot()
'''plt.title("Seasonality - Time Series spx", size=24)'''
plt.show()
'''
For this dataset, also using multiplicative approach there is evidence of seasonality.

This means that trend follows actual data closely next.
'''