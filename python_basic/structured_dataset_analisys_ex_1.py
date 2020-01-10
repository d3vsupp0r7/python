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
print(' 2.1) Using numpy for mapping ordinal_categorical_values => to  =>  business_custom_mapping')
'''
-> IMPO: import dataset using index_col=0
-) use of vectorize
'''
pd_dataset = pd.read_csv(ds_filename,index_col=0)
X=pd_dataset.values
print(' Origin DS')
print(X)
fmap = np.vectorize(lambda t:size_mapping[t])
X[:,0] = fmap(X[:,0])
print(' Mapped DS')
print(X)

print('-) Manage nominal variables')
print('     ** ONE-HOT ENCODING **')
print('       -) ONE-HOT ENCODING using pandas')
column_to_onehotencoding="color"
ds_one_hot_encoding = pd.get_dummies(pd_dataset,columns=[column_to_onehotencoding])
print(' Origin DS')
print(pd_dataset.head())
print(' OneHotEncoding DS')
print(ds_one_hot_encoding.head())
print('       -) ONE-HOT ENCODING using numpy array')
'''
To operate on numpy array manipulation for ONE-HOT ENCODING operation we need to import sklearn libraries
'''
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

print('       -) ONE-HOT ENCODING using numpy array')
print('          -) Use of LabelEncoder for Ordered Quality Variable')
le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
print(' Origin DS')
print(pd_dataset.head())
print(' Modified DS')
print(X[:5])

print('          -) Use of OneHotEncoder for UnOrdered Quality Variable')
enc = OneHotEncoder(categorical_features=[1])
'''
Math note: the encoder will produce a sparse matrix
'''
X_sparse = enc.fit_transform(X)
X = X_sparse.toarray()
print(X[:5])

# Example: Use of sklearn 0.22 and so on (most recent version) -> TODO: Make more clear this section
from sklearn.compose import ColumnTransformer
print('          -) Use of ColumnTransformer')
columnTransformer = ColumnTransformer([('size', OneHotEncoder(), [1])], remainder='passthrough')
out = columnTransformer.fit_transform(pd_dataset)
print(out)

print('*** MANAGE MISSING DATA ***')
ds_nan_filename = "pandas_dataset_examples/iris_nan.csv"
pd_ds_nan_filename = pd.read_csv(ds_nan_filename)
print('   DS shape ')
print(pd_ds_nan_filename.shape)

feature_1 = "class"
Y_numpy_array = pd_ds_nan_filename[feature_1].values
X_numpy_array = pd_ds_nan_filename.drop(feature_1,axis=1).values

print('   -) Y ')
print(Y_numpy_array[:5])
print('   -) X ')
print(X_numpy_array[:5])

print('   -) Remove row or columns with nan values. function dropna() ')
'''
The dropna will be used carefully, because 
'''
print('         Remove rows nan ')
ds_with_drop = pd_ds_nan_filename.dropna()
print('   DS drop shape  ')
print(ds_with_drop.shape)
print('   DS  Origin')
print(pd_ds_nan_filename.head())
print('   DS  with no nan')
print(ds_with_drop.head())
print('         Remove cols nan ')
print('   DS  Origin')
print(pd_ds_nan_filename.head())
ds_cols_with_drop = pd_ds_nan_filename.dropna(axis=1)
print('   DS  with no nan - cols')
print(ds_cols_with_drop.head())

print('*** MANAGE MISSING DATA: Imputation approach ***')
print('   DS  Origin')
print(pd_ds_nan_filename.head())
print('   DS  Imputation with mean')
mean_for_nan_replace = pd_ds_nan_filename.mean()
ds_with_mean = pd_ds_nan_filename.fillna(mean_for_nan_replace)
print(ds_with_mean.head())
print('   DS  Imputation with mode')
'''
The mode will return a ordered dataframe
'''
mode_for_nan_replace = pd_ds_nan_filename.mode().iloc[0]
ds_with_mode = pd_ds_nan_filename.fillna(mode_for_nan_replace)
print(ds_with_mode.head())

## Using numpy vector ##
# TODO: Imputer deprecated in sklearn 0.20 => Update this section
print('-) Imputation with sklearn')
from sklearn.preprocessing import Imputer

X_DS = pd_ds_nan_filename.drop("class", axis=1)

print('-) Imputation with sklearn: mean')
imp = Imputer(strategy="mean",axis=0,missing_values="NaN")
imputation = imp.fit_transform(X_DS)
print(imputation[:5])

print('-) Imputation with sklearn: median')
imp = Imputer(strategy="median",axis=0,missing_values="NaN")
imputation = imp.fit_transform(X_DS)
print(imputation[:5])

print('-) Imputation with sklearn: most_frequent')
imp = Imputer(strategy="most_frequent",axis=0,missing_values="NaN")
imputation = imp.fit_transform(X_DS)
print(imputation[:5])

print('*** MAKE DATASET ON SAME SCALE VALUES ***')
'''
This example analyze the wine dataset with alcol and flavonoids.
This example will focus on normalization and standardization of dataset based on a very high different scales of 
values of this two features:
-) alcol
-) flavonoids

Comparing the min and max of these features with describe() method, we will observe the
high difference of values. This observation will probaly means that we need to normalize
these data to a common scale of values.
'''
ds_filename = "pandas_dataset_examples/wine.data"
pd_ds = pd.read_csv(ds_filename,usecols=[0,1,7],names=['class','alcol','flavanoids'])
print(' Origin DS')
print(pd_ds.head())

ps_ds_copy = pd_ds.copy()
features_name_array = ["alcol", "flavanoids"]
to_norm = pd_ds[features_name_array]

feature_1 = 'class'
print(' Features ')
features_numpy_array = pd_ds[feature_1].values
print(features_numpy_array[:5])

print(' Features Data')
feature_data_numpy_array = pd_ds.drop(feature_1,axis=1).values
print(feature_data_numpy_array[:5])

print(' DS analisys')
print(pd_ds.describe())

ps_ds_norm = ps_ds_copy[features_name_array] = (to_norm - to_norm.min()) / (to_norm.max() - to_norm.min())

print(' DS normalized')
print(ps_ds_norm.head())

print("*** Normalize numpy array with sklearn")
from sklearn.preprocessing import MinMaxScaler
print(' -) Use of MinMaxScaler')

mms = MinMaxScaler()
X_norm = feature_data_numpy_array.copy()
X_norm = mms.fit_transform(X_norm)
print(X_norm[:5])

print("*** Standardization numpy array with sklearn")
X_std = pd_ds.copy()
to_std = X_std[features_name_array]
X_std[features_name_array] = (to_std-to_std.mean())/to_std.std()
print(X_std[:5])

print("*** Standardization numpy array with sklearn: StandardScaler")
from sklearn.preprocessing import StandardScaler

X_sklearn_std = feature_data_numpy_array.copy()
ss = StandardScaler()