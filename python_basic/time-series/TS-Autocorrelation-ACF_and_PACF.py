import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose


#
ds_filename = "../time-series_datasets/market_index_2018.csv"

# Define the complete dataframe
pd_dataset_compl = pd.read_csv(ds_filename)
# working dataset: pd_dataset
pd_dataset = pd.read_csv(ds_filename)

#
df_comp = pd_dataset.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst=True)
df_comp.set_index("date", inplace=True)
df_comp = df_comp.asfreq('b')
df_comp = df_comp.fillna(method='ffill')
#
df_comp['market_value'] = df_comp.spx

# Dickey-Fuller test
print('** Stationality analisys **')
print('** Dickey-Fuller Test **')
print(sts.adfuller(df_comp.market_value))

del df_comp['spx']
del df_comp['dax']
del df_comp['ftse']
del df_comp['nikkei']
size = int(len(df_comp) * 0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]

# Seasonality decomposition
s_dec = seasonal_decompose(df.market_value)
print(s_dec)
# Additive
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
# Multiplicative approach
s_dec_multiplicative = seasonal_decompose(df.market_value, model="multiplicative")
s_dec_multiplicative.plot()
'''plt.title("Seasonality - Time Series spx", size=24)'''
plt.show()
'''
For this dataset, also using multiplicative approach there is evidence of seasonality.

This means that trend follows actual data closely next.
'''
''' # Autoregression
lags=40  means we analize the last 40 periods before the current one.
40 lags analisys is the reccomandaded base start analisys because more langs analisys means
a lot of calculation, so this parameter ca be adjusted accordling to our needs.

zero parameters indicate if we eant to include current period values into graph.
For correlation graph this will be unnecessary beacuse the correlation with a value and itself is always one.

'''
sgt.plot_acf(df.market_value,lags=40, zero=False)
plt.title("ACF S&P", size=24)
plt.show()
'''
Explanation of autocorrelation graph.

On the X axis we have lags.
On Y axis we have the possible values for autocorrelation coefficient.

Correlation can only take values from range -1 -> 1 included.

The line across the plot represents the autocorrelation between time series and a lagged copy of itself.
Reading the graph from left to right, the first line  indicate autocorrelation one time period ago, the second line
represent the coefficient value for two periods ago.. and so on.

The blu area around the x asis represent "Significance".
The values outside this area, are significantly different from 0, this suggest the
existence of autocorrelation for that specific lag.

Grater distance in time, the more unlikely it is that this auto correlation persist.
We need to make sure the autocorrelation coefficient in higher lag is sufficiently greater
to be significatively different from zero.

If teh top element of line are significant and off the blue region, this suggest the coefficient are significant
this is and indicator of time dependence in the data.  

Another point is that autocorrelation dimishes as the lags increase.
This means that prices of previous month back can serve as decent estimates.
'''

print('Analisys of ACF into a withe noise time series')
wn = np.random.normal(loc = df.market_value.mean(), scale = df.market_value.std(), size = len(df))
#
df['wn'] = wn
print(df.describe())
#
sgt.plot_acf(df.wn,lags=40, zero=False)
plt.title("ACF S&P - WhiteNoise Example", size=24)
plt.show()
'''
Analisys of Autocorrelationgraph of a withe noise time series

1) We have both positive and negative autocorrelation
2) All the lineas are in the blue area, this means that coefficients are not
significative across all plot. This suggest there is no autocorrelation
for any lag. This can be an indicator of withe noise time series.

'''

#Partial autocorrelation - PACF
## PACT OLS method (Order Of Least Square)
sgt.plot_pacf(df.market_value,lags=40, zero=False,method="ols")
plt.title("PACF S&P - OLS Method", size=24)
plt.show()
'''
Analisys of PACF OLS Graph

PACF cancel all additional channels whicha previous period value effects the present one.

The first evaluation of PACF and ACF is identical because no previous other channel
affect present values.
'''