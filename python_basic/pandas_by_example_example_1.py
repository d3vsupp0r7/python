from itertools import combinations

import numpy as np
import pandas as pd

###
from dateutil.parser import _resultbase
from setuptools.command.saveopts import saveopts

ds_filename = "pandas_dataset_examples/iris.csv"
categorical_variable = "species"
###
# Working with files
print('*** Pandas: Working with CSV files ***')

## dataset loading
pd_dataset = pd.read_csv(ds_filename)
print('** Datataset loaded = ' + ds_filename)

print('** Datataset HEAD info ')
print(pd_dataset.head())
print('** Datataset TAIL info ')
print(pd_dataset.tail())

print('** Datataset Basic statistic info ')
print(pd_dataset.describe())

print('** Datataset: Unique Label categorical variable values')
categorical_feature_species = pd_dataset[categorical_variable].unique()
print(categorical_feature_species)

print('** STEP 1: Normalization numerical variables')
# 1.1 Drop categorical variables
working_ds = pd_dataset.drop(categorical_variable,axis=1)
normalization_dividend = (working_ds - working_ds.min() )
normalization_divisor = (working_ds.max() - working_ds.min())
working_ds_normalization = (normalization_dividend / normalization_divisor)

print(' -) Normalized dataset ')
print(working_ds_normalization.head())

print(' -) Sort a dataset by column ')
column_name = 'petal_length'
print('     * Origin Dataset ')
print( pd_dataset.sort_values(column_name).head() )
print('     * Normalized Dataset ')
print( working_ds_normalization.sort_values(column_name).head() )

print(' -) Use of groupBy on observations for Dataset ')
print('   Example of groupBy on categorical variables ')
group_by_category_var = pd_dataset.groupby(categorical_variable)
print('    Print the mean on groupby function result')
print(group_by_category_var.mean())


## Use of NumPy ##
print('*** USE OF NumPy Library ***')
import numpy as np
print(' -) Apply functions on row or columns => function: apply() ')
print('     -) Apply functions on rows')
# Count no 0 values on rows
result = pd_dataset.apply(np.count_nonzero, axis=1).head()
print(result)
# Count no 0 values on cols
print('     -) Apply functions on Cols')
result = pd_dataset.apply(np.count_nonzero, axis=0).head()
print(result)

print(' -) Apply functions values by values => function: applymap() ')
print('      round values to next near integer')
# lambda function: is a function to use one time
working_ds_round_next_int = working_ds.applymap(lambda val:int(round(val,0)))
print('      Origin ')
print(working_ds.head())
print('      Rounded out ')
print(working_ds_round_next_int.head())

print('** Manage invalid values **')
working_ds_nan = working_ds.copy(0)
sample_data = np.random.randint(working_ds.shape[0], size=10)
print(' Generated sample data')
print(sample_data)

# set sample data to nan_ds
print(' Generated None values')
column_name = "petal_length"
working_ds_nan.loc[sample_data,column_name] = None
# Sum number of elements that are null into column
result = working_ds_nan[column_name].isnull().sum()
print('Sum of null values: ' + str(result) )

print(' -) Manage invalid values => fillna() ')
mean_petal_length = working_ds_nan[column_name].mean()
print('     Mean:  ' + str(mean_petal_length) )
'''
Using only fillna() will not produce modify on original set.
To permanently modify a fillna() operation we can:

1) Assign the fillna() output to a variable, as example:
result = working_ds_nan[column_name].fillna(mean_petal_length)

2) Use inplace parameter
'''
print(' Sample 1: Using fillna() with assign')
result = working_ds_nan[column_name].fillna(mean_petal_length)
print(result.head())

print(' Sample 2: Using fillna() with inplace')
print('     Sample 2 - Origin DS')
print(working_ds_nan.head())
working_ds_nan[column_name].fillna(mean_petal_length, inplace=True)
print('     Sample 2 with inplace')
print(working_ds_nan.head())

## Use of matplotlib ##
print('*** USE OF matplotlib Library ***')
print('   -)  matplotlib.pyplot ')
import matplotlib.pyplot as plt
print(' -)  Original dataset plotting')
x_axis_col_name = 'sepal_length'
y_axis_col_name = 'sepal_width'
print('     -) Plot as scatter plot')
pd_dataset.plot(x=x_axis_col_name, y=y_axis_col_name,kind='scatter')
plt.show()