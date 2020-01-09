from asynchat import simple_producer

import pandas as pd
import numpy as np

ds_filename = "pandas_dataset_examples/shirts_structured_ds.csv"

## dataset loading
pd_dataset = pd.read_csv(ds_filename)
print('** Datataset loaded = ' + ds_filename)

print('** Datataset HEAD info ')
print(pd_dataset.head())
print('** Datataset TAIL info ')
print(pd_dataset.tail())

print('-) Get rows of dataset as numpy array')
ds_numpy_array = pd_dataset.values
print(ds_numpy_array)

print('-) Manage ordinal variables')
# Step 1: define a python dictionary that contains the mapping
size_mapping = {"S":0,"M":1,"L":2,"XL":3}
print(' 1) Define python dictionary for mapping ')
print(size_mapping)
print(' 2) Using pandas for mapping ordinal_categorical_values => to  =>  business_custom_mapping')
ordinal_feature = "size"
print('  Original DS  ')
print(pd_dataset.head())
pd_dataset[ordinal_feature] = pd_dataset[ordinal_feature].map(size_mapping)
print('  Mapped DS  ')
print(pd_dataset.head())


print('-) Manage nominal variables')