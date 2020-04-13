import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#
ds_filename = "../time-series_datasets/market_index_2018.csv"

#Define the complete dataframe
pd_dataset_compl = pd.read_csv(ds_filename)
# working dataset: pd_dataset
pd_dataset = pd.read_csv(ds_filename)

print('*) DS Info')
print(pd_dataset.info())
print('-) DS shape')
print(pd_dataset.shape)

'''
Remember:
The describe method only get statistics for numeric variables. This mean that 
no information are given for date field.

Column meanings:
count: count the number of observation for each column

'''
print('-) DS describe()')
print(pd_dataset.describe())

'''
IMPO: When analysing at first glance the output of describe method, we need to make attention if values
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
Inside the python analysis of time series often the date field is used as index.
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
Also, we can use this analysis on the single column of pandas dataframe, like:
panda_ds_object.columnname.isna().sum()
'''
print(pd_dataset.isna().sum() )
print(' -) Using only column name')
print(pd_dataset.spx.isna().sum() )

# Plotting timeseries data
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

# Plot two timeseries together
print(pd_dataset_compl.spx.plot(figsize=(20,5)) )
print(pd_dataset_compl.ftse.plot(figsize=(20,5)) )
plt.title("S&P 500 vs FTSE 100")
plt.show()

# QQ Plot: Quantile-Quantile Plot
'''
The Quantile-Quantile plot ia a tool used in analytics to determinate whether a data set is distributed a certain way 
unless specified, otherwise the QQ plot showcases how data fits a normal distribution.
'''
import scipy.stats
import pylab

scipy.stats.probplot(pd_dataset_compl.spx, plot = pylab)
pylab.title('Quantile-Quantile Plot for S&P 500 index')
pylab.show()
#
scipy.stats.probplot(pd_dataset_compl.ftse, plot = pylab)
pylab.title('Quantile-Quantile Plot for FTSE 100 index')
pylab.show()
#
scipy.stats.probplot(pd_dataset_compl.dax, plot = pylab)
pylab.title('Quantile-Quantile Plot for DAX 30 index')
pylab.show()
#
scipy.stats.probplot(pd_dataset_compl.nikkei, plot = pylab)
pylab.title('Quantile-Quantile Plot for Nikkei 225 index')
pylab.show()
'''
QQPlot info:
The QQ Plot, takes all the values a variable can take and arranges them in ascending order.

The Y axis represent teh price with the highest one at the top and the lowest at bottom.
The X axis represent the theoretical quintiles of the data set and how many standard deviations away 
from the mean these values are.

The red line represent what the data points should follow if they are normally distributed. This is the normal behavior
for time series data.
'''

## Trasforming DataFrame to TimeSeries  ##
print('*** Transforming DataFrame to TimeSeries ***')
'''

'''
print('-) DS describe() with data')
print(pd_dataset.date.describe())
'''
NOTES: Using pd_dataset.date.describe(), by default python give to all "date" fields the value 1. So, any single 
data holds a "top" value and the method randomly select one to display. 
'''
# Transforming data filed to DateTime type that libraries can recognize as data field.
'''
So for transforming data filed to DateTime type we do that transformation, using the pandas library, using the method 
*to_datetime* that accept as parameter the column of dataset that represent
the data field.
'''
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
## Check if transformation of string to date is executed
print('-) DS describe() with data AFTER MODIFICATION')
print(pd_dataset_compl.date.describe())
print(pd_dataset_compl.describe())
'''
IMPORTANT NOTES: Pandas add to data also the time after the conversation of date field. This is helpful in order to
manage the date field in it's complexity.
Once the "data" field is transformed to be managed properly, this field can be used as index to execute the various
operations.

Each value in our row should correspond to a time period. This is important because we want to
examine specific chunks of data between two concrete dates. So "data" field of dataset can be seen as a list of 
potential cut-off points. This can be achieved setting "date" field as a index.
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

This because once a "date" field became an index, we no longer save its values as a separate 
attribute in the data frame.

'''
# This line give ERROR: see Notes print(pd_dataset_compl.date.describe())
## Set a frequency to trasform dataset into time series ##
'''
The pandas library allow us to assign the frequency of set using it's method *asfreq()*  on dataframe object.
The method can use different parameters:
-) h => hours
-) w => weekly
-) d => daily
-) m => monthly
-) a => annual
-) b => includes only business day (no saturdays and sundays)

The parameter is mandatory!.

If some date point are not present, the library will automatically add the missing days/period and insert NaN values
on each rows and we generated new periods witch do not have values associated with them.
'''
pd_dataset_compl = pd_dataset_compl.asfreq('d')
print('-) DS head() AFTER asfreq() MODIFICATION')
print(pd_dataset_compl.head())

pd_dataset_compl = pd_dataset_compl.asfreq('b')
print('-) DS head() AFTER asfreq() MODIFICATION for business days')
print(pd_dataset_compl.head())

## Managing the missing values  ##
print('*** TimeSeries: Managing the missing values ***')
'''
As basic, we use the *isna()* on dataframe to know if there are some null values.
'''
print('-) DS isna() on original dataframe')
print(pd_dataset.isna().sum() )
print('-) DS isna() on dataset AFTER asfreq() MODIFICATION for business days ')
print(pd_dataset_compl.isna().sum() )
'''
IMPO NOTES: This means that changing frequency can produce a MISSING DATA when we analyze the dataset.
These mean that we need to fill these missing values.
'''
print('-) DS filling missing values using the *fillna()* method')
'''
*fillna()* method fill missing data with different approach:

1) Front filling: Assign the value of previous period;

2) Back filling: Assign the value of next period;

3) Assign the same value to all missing periods. This mean that for the missing values,we assign to it
the average of all values of time-series analyzed into file.

'''
# Example: fillna with front filling approach
pd_dataset_compl.spx = pd_dataset_compl.spx.fillna(method="ffill")
print(pd_dataset_compl.spx);
print('-) ffill test on spx column')
print(pd_dataset_compl.isna().sum() )

# Example: fillna with back filling approach
pd_dataset_compl.ftse = pd_dataset_compl.ftse.fillna(method="bfill")
print('-) bfill test on ftse column')
print(pd_dataset_compl.isna().sum() )

# Example: fillna with average approach
pd_dataset_compl.dax = pd_dataset_compl.dax.fillna(value=pd_dataset_compl.dax.mean())
print('-) average filling test on dax column')
print(pd_dataset_compl.isna().sum() )

## Managing the dataframe: Operations on columns  ##
print('*** TimeSeries: Managing the dataframe with operations on columns ***')
'''
Example of use: We want to analyze only a specific index. There are are two essential reason to do that:
1) Managing a dataset with less data means that we are more fast to manipulate the entire data frame.
2) Clarity, easier to keep track of the dataset.
'''
## Add a new column of the fly to dataset
print('-) Add the new column: market_value: market_value is spx values')
pd_dataset_compl['market_value'] = pd_dataset_compl.spx
print(pd_dataset_compl.describe())

print('-) Delete unused column')
del pd_dataset_compl['spx'],pd_dataset_compl['ftse'], pd_dataset_compl['dax'],pd_dataset_compl['nikkei']
print(pd_dataset_compl.describe())

## Timeseries: Splitting data for model: Train/Test set  ##
print('*** Timeseries: Splitting data for model: Train/Test set ***')
'''
NOTES: In order to apply a machine learning model to time series, we need to split our data to two dataset,
the train and test data. splitting the data into this two approach is used to compare prediction ( model trained 
on train dataset) to actual values (test dataset values). Obviusly the colesr the forecasts, the better the model.

For a time series approach is not possible to shufffle data form Train and test dataset.
-) The training ses is from a beginning up to some cut off point.
-) The testing set is from the cutoff point until the end.

Define the cutoff point is the problem.
*) If too large, the model will fit the training set too well and perform poorly with a new dataset
*) If too small we can't create an accurate model.

A basic compromise, we can split with 80/20 rule. 80% Traning / 20% Test .
After the split is a good practice to verify if the split was succesfully executed  with comparing the 
training dataset tail() values to test dataset head() to verify if some values are overlapped and manage it
as needed.

'''
print('-) Use of *iloc()* method to split the data')
# Step one: Determinate the cutoff point
'''

'''
print('-) Global dataset size: ' + str(len(pd_dataset_compl)))
size = int(len(pd_dataset_compl)*0.8)
print('-) 0.8 size: ' + str(size)) 

# Step two: create training dataset
training_df = pd_dataset_compl.iloc[:size]
test_df = pd_dataset_compl.iloc[size:]

print('-) Verify training dataframe splitting')
print('   -) Verify training dataframe splitting (TAIL)')
print(training_df.tail())
print('-) Verify test dataframe splitting')
print('   -) Verify test dataframe splitting (HEAD)')
print(test_df.head())
print('    -) Verify test dataframe splitting (TAIL)')
print(test_df.tail())