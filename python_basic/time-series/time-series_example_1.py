import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#
ds_filename = "../time-series_datasets/market_index_22018.csv"

#Define the complete dataframe
pd_dataset_compl = pd.read_csv(ds_filename)
# working dataset: pd_dataset
pd_dataset = pd.read_csv(ds_filename)

print('*) DS Info')
print(pd_dataset.info())
print('-) DS shape')
print(pd_dataset.shape)

'''
Rememeber:
The describe method only get statistics for numeric variables. This mean that 
no information are given for date field.

Clolumn meanings:
count: count the number of observation for each column

'''
print('-) DS describe()')
print(pd_dataset.describe())

'''
IMPO: When analizing at first glance the output of describe method, we nedd to make attention if values
into result are an huge difference in magnitude.
As example, spx has the minimum more lowest respect to other market indexes MIN values.

'''

print('-) DS sample data')
print(pd_dataset.head())
print('-) DS correlation information')
from tabulate import tabulate
print(tabulate(pd_dataset.corr()))

# Exploring the dataset
'''
Inside the python analisys of time series often the date field is used as index.
This example uses four market index (used in stock exchange):
-) S&P 500    -> spx        USA
-) DAX 30     -> dax        GERMANY
-) FTSE 100   -> ftse       LONDON UK
-) NIKKEI 225 -> nikkei     JAPAN

-) The number indicates the number of companies included into portfolio.
As example
S&P 500 : Includes 500 companies
DAX 30  : Includes 30 companies
'''

'''
Notes for data processing
'''
print('-) DS information: There are some nulls values ? ')
print(pd_dataset.isna())
'''
TIP: Simple way to check null values: panda_ds_object.isna().sum()
If some row have value different from 0, there are null values inside that column.
Also, we can use this analisys on the single column of pandas dataframe, like:
panda_ds_object.columnname.isna().sum()
'''
print(pd_dataset.isna().sum() )
print(' -) Using only column name')
print(pd_dataset.spx.isna().sum() )

# Ploting timeseries data
'''

'''
print(pd_dataset_compl.spx.plot(title="S&P 500 Prices"))
plt.show()

print(pd_dataset_compl.spx.plot(figsize=(20,5), title="S&P 500 Prices - Better visual"))
plt.show()

print(pd_dataset_compl.dax.plot(figsize=(20,5), title="DAX 30 Prices - Better visual"))
plt.show()

print(pd_dataset_compl.ftse.plot(figsize=(20,5), title="FTSE 100 Prices - Better visual"))
plt.show()

print(pd_dataset_compl.nikkei.plot(figsize=(20,5), title="Nikkei 225 Prices - Better visual"))
plt.show()

# Plot two timeseries togheter
print(pd_dataset_compl.spx.plot(figsize=(20,5)) )
print(pd_dataset_compl.ftse.plot(figsize=(20,5)) )
plt.title("S&P 500 vs FTSE 100")
plt.show()

# QQ Plot: Quantile-Quantile Plot
'''
The Quantile-Quantile plot ia a tool used in analytics to determinate whether a data set is distributed a certain way unless
specified, otherwise the QQ plot showcases how data fits a normal distribution.
'''
import scipy.stats
import pylab

scipy.stats.probplot(pd_dataset_compl.spx, plot = pylab)
pylab.show()
'''
QQPlot info:
The QQ Plot, takes all the values a variable can take and arranges them in ascending order.

The Y axis represent teh price with the highest one at the top and the lowest at bottom.
The X axis represent the theorical quintiles of the data set and how many standard deviations away 
from the mean these values are.

The red line represent what the data points should follow if they are normally distributed. This is the normal behavoir
for time series data.
'''

## Trasforming DataFrame to TimeSeries  ##
print('*** Trasforming DataFrame to TimeSeries ***')
'''

'''
print('-) DS describe() with data')
print(pd_dataset.date.describe())
'''
NOTES: Using pd_dataset.date.describe(), by default python give to all "date" fields the value 1. So, any single 
data holds a "top" vale and the method randomly select one to display. 
'''
# Trasforming data filed to DateTime type thazth libraries can recognize as data field.
dataframe_with_data_example_1 = pd.to_datetime(pd_dataset_compl.date)
'''
By default, the library assumes that date are in format mm/dd/yyyy, to convert it into format dd/mm/yyyy
we ned to pass the additional parameter dayfirst
'''
dataframe_with_data_example_2 = pd.to_datetime(pd_dataset_compl.date, dayfirst=True)
##
print('-) DS head() invocation')
print(pd_dataset_compl.head())
pd_dataset_compl.date = pd.to_datetime(pd_dataset_compl.date, dayfirst=True)
print('-) DS head() invocation with data field processed AFTER MODIFICATION')
print(pd_dataset_compl.head())
## Check if trasformation of string to date is executed
print('-) DS describe() with data AFTER MODIFICATION')
print(pd_dataset_compl.date.describe())
print(pd_dataset_compl.describe())
'''
IMPORTANT NOTES: Pandas add to data also the time after the conversation of date field. This is helpful in order to
manage the date field in it's complexity.
Once the "data" field is trasformed to be managed properly, this field can be used as index to execute the various
operations.

Each value in our row should correspond to a time period. This is important because we want to
examine specific chunks of data between two concrete dates. So "data" field of dataset can be seen as a list of 
potential cut-off points. This can be acheived setting "date" field as a index.
'''
# setting date field as a index
pd_dataset_compl.set_index("date", inplace=True)
print('-) DS head() invocation with data field processed AFTER INDEX MODIFICATION')
print(pd_dataset_compl.head())
'''
IMPO NOTES: We see from console that the new "data" field is become a index, because we not see indexes at 
left side of head() method printing.
If we use the date.describe() on dataset we have the following error
** AttributeError: 'DataFrame' object has no attribute 'date' **
[
print(pd_dataset_compl.date.describe())
will give
** AttributeError: 'DataFrame' object has no attribute 'date' **
]

This beacause once a "date" field becaome an idex, we no longer save its values as a separate 
attribute in the data frame.

'''
# This line give ERROR: see Notes print(pd_dataset_compl.date.describe())
## ##
