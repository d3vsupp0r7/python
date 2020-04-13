import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose

#
ds_filename = "../time-series_datasets/market_index_2018_RandWalk_example.csv"

#Define the complete dataframe
pd_dataset_compl = pd.read_csv(ds_filename)
# working dataset: pd_dataset
pd_dataset = pd.read_csv(ds_filename)

# Working with dataset
rw=pd_dataset.copy()
rw.date = pd.to_datetime(rw.date, dayfirst = True)
rw.set_index("date", inplace=True)
rw=rw.asfreq('b')
#
size = int(len(rw)*0.8)

print("rw describe()")
print(rw.describe())
print("rw head()")
print(rw.head())
#
