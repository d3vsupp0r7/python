"""
Example of dataset manipulation.
-) Manage ordinal variables
-) Use of LabelEncoding and OneHotEncoding
-) Manage missing data
    -) Imputation example
-) Normalization and Standardization Example
-) Dataset splitting using slearn and pandas
"""

import numpy as np
import pandas as pd
import sklearn

print('The pandas version is {}.'.format(pd.__version__))
print('The numpy version is {}.'.format(np.__version__))

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
To operate on numpy array manipulation for ONE-HOT ENCODING operation we need to import sklearn libraries.
IMPO: This code working with sklearn version: 0.21.3.
'''
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

print('The scikit-learn version is {}.'.format(sklearn.__version__))

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
X_sklearn_std = ss.fit_transform(X_sklearn_std)
print(X_sklearn_std[:5])

print('*** DATASET SPLITTING ***')
'''
Example of boston house dataset from sklearn library.
The sklearn loads it's dataset into dictionary form.
'''
from sklearn.datasets import load_boston
sk_ds = load_boston()
print("-) DS info")
print(sk_ds.DESCR)
print("     Shape")
print(sk_ds.data.shape)
print(" Get features name")
print(sk_ds.feature_names)

# Convert sklearn dataset to pandas
print("-) Convert sklearn dataset to pandas dataframe")
pd_ds = pd.DataFrame(sk_ds.data)
print(pd_ds.head())
pd_ds.columns = sk_ds.feature_names
print(pd_ds.head())

# Convert sklearn dataset to numpy
print('Obtaining data')
sk_ds_data_numpy = sk_ds.data
print('Shape: ' + str(sk_ds_data_numpy.shape) )
print(sk_ds_data_numpy)
print('Obtaining Target  data')
sk_ds_target_numpy = sk_ds.target
print(sk_ds_target_numpy)

# Dataset division using sklearn
from sklearn.model_selection import train_test_split
# Arrays that contains the splitted dataset
'''
Notes: The suddivision from train and test set is exposed as fraction on % of splitting passed as 
argument of *test_size* parameter. 
'''
X_train, X_test, Y_train, Y_test = train_test_split(sk_ds_data_numpy,sk_ds_target_numpy,test_size=0.3)
print('DS subdivision: ')
print('X_train: ' + str(X_train.shape) )
print('X_test:  ' + str(X_test.shape) )
print('Y_train: ' + str(Y_train.shape) )
print('Y_test:  ' + str(Y_test.shape) )

'''
Splitting dataset using pandas.
'''
PD_sk_to_pd_ds = pd_ds.copy()
# This instruction will randomize the original dataset and assign it's 70% to pandas object: train_set.
# IMPO: random_state=0 is important to have constant iteration and same splitting over different iterations
train_set = PD_sk_to_pd_ds.sample(frac=0.7, random_state=0)
# This istruction will assign the remains data (30% to test_set)
test_set = PD_sk_to_pd_ds.drop(train_set.index)

print ('Training set')
print (train_set.shape)
print (train_set.head())
print ('\nTest set')
print (test_set.shape)
print (test_set.head())
print ('\nOriginal DataFrame')
print (PD_sk_to_pd_ds.head())