import numpy as np
import pandas as pd

###
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
print(pd_dataset.head())

print('** Datataset Basic statistic info ')
print(pd_dataset.describe())

print('** Datataset: Unique Label categorical variable values')
categorical_feature_species = pd_dataset[categorical_variable].unique()
print(categorical_feature_species)

print('** STEP 1: Normalization numerical variables')
working_ds = pd_dataset.drop(categorical_variable,axis=1 )