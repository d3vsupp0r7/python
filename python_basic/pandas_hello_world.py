from operator import irshift

import numpy as np
import pandas as pd

'''
Series:
Series is a one-dimensional array, which can hold any type of data, 
such as integers, floats, strings, and Python objects too.
'''
seriesDataExample = pd.Series(np.random.randn(5))
print(seriesDataExample)
'''
The series function creates a pandas series that consists of an index, which is the first column, and the second column consists of random values.
'''
seriesDataExampleWithLabel =pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
print(seriesDataExampleWithLabel)

d = {'A': 10, 'B': 20, 'C': 30}
dExample = pd.Series(d)
print(dExample)

####################################################
# DataFrame
'''
DataFrame is a 2D data structure with columns that can be of different datatypes. 
It can be seen as a table. 
A DataFrame can be formed from the following data structures:
    -) A NumPy array
    -) Lists
    -) Dicts
    -) Series
    -) A 2D NumPy array
'''
## Dataframe creation
dataFrameData1 = {'c1': pd.Series(['A', 'B', 'C']),
     'c2': pd.Series([1, 2., 3., 4.])}
dataFrameObject1 = pd.DataFrame(dataFrameData1)
print(dataFrameObject1)

dataFrameData2 = {'c1': ['A', 'B', 'C', 'D'],
                  'c2': [1, 2.0, 3.0, 4.0]}
dataFrameObject2 = pd.DataFrame(dataFrameData2)
print(dataFrameObject2)

#print("First row: " + dataFrameObject1.head())
## Read first row of dataFrame

# Panel
'''
A Panel is a data structure that handles 3D data.
'''


# Working with files
print('*** Pandas: Working with CSV files ***')
## Iris dataset
iris_pd_dataset = pd.read_csv("pandas_dataset_examples/iris.csv")
print('** Pandas: HEAD')
print('Using head as attribute will print all dataset')
print(iris_pd_dataset.head)
print('Show First 5 row (index 0->4).')
print('REMEMBER: All row are numbered starting from 0 indexing: ')
print(iris_pd_dataset.head())
print('Show First 10 row')
print(iris_pd_dataset.head(10))
print('** Pandas: TAIL')
print('Show Last 5 row (index 0->4).')
print(iris_pd_dataset.tail())
print('Show Last 10 row.')
print(iris_pd_dataset.tail(10))

print('** Pandas: CSV With no header')
iris_pd_ds_no_header = pd.read_csv("pandas_dataset_examples/iris_noheader.csv")
print(iris_pd_ds_no_header.head())
print('Manually add header')
print(' -) Manually add header: Headers < num_col')
iris_pd_ds_no_header = pd.read_csv("pandas_dataset_examples/iris_noheader.csv", header=None, names=["h1","h2","h3","h4"])
print(iris_pd_ds_no_header.head())
print(' -) Manually add header: Headers = num_col')
iris_pd_ds_no_header = pd.read_csv("pandas_dataset_examples/iris_noheader.csv", header=None, names=["h1","h2","h3","h4","h5"])
print(iris_pd_ds_no_header.head())
print(' -) Manually add header: Headers > num_col')
iris_pd_ds_no_header = pd.read_csv("pandas_dataset_examples/iris_noheader.csv", header=None, names=["h1","h2","h3","h4","h5","h6"])
print(iris_pd_ds_no_header.head())

print('** Pandas: Columns')
print(iris_pd_dataset.columns)

print('** Pandas: Info on columns type of dataset')
print(' -) info() as a function -> get type of columns')
print(iris_pd_dataset.info())
print(' -) info as a attribute -> print a dataset')
print(iris_pd_dataset.info)

print('** Pandas: Working with datasets')
print("Iris dataset dimension info: " + str(iris_pd_dataset.shape) )
# Select only one column from dataset
Y = iris_pd_dataset['sepal_length']

print(' -) Get all column values of: sepal_length')
Y =  iris_pd_dataset['sepal_length']
print(Y)
print(' -) Get all column values of: sepal_length.\n Col name as string variable.')
col_name = "sepal_length";
Y =  iris_pd_dataset[col_name]
print(Y.head())
print(' -) Get subset of columns.')
Y_subset_cols = iris_pd_dataset[["sepal_length","sepal_width","petal_length"]]
print(Y_subset_cols.head())
col_array = ["sepal_length","sepal_width","petal_length"]
print(' -) Get subset of columns.\n Col names as string array.')
Y_subset_cols_str = iris_pd_dataset[col_array]
print(Y_subset_cols_str.head())
print(' *) NOTE IF WE SELECT A SUBSET OF COLUMNS, IT''S TYPE. IS A DataFrame')

# A Single column of dataset in panda is a "Series" object
print(' -) Get types of objects')
print('Y is of type: ' + str(type(Y)) )
print('Y_subset_cols is of type: ' + str(type(Y_subset_cols)) )
print('iris_pd_dataset is of type: ' + str(type(iris_pd_dataset)) )

print(' -) REMOVE A COLUMN From Dataset: drop()')
# drop require the axis parameter
# IMPO: The axis parameter means axis=0 refers to rows, axis=1 refers to columns
Y_col_rem = iris_pd_dataset.drop("species", axis=1)
print(Y_col_rem.head())

print(' -) Remove a row by index. Remove the first row: index[0]')
Y_drop_sample = iris_pd_dataset.head()
print('ORIGINAL DATASET')
print(Y_drop_sample)
drop_out = Y_drop_sample.drop(Y_drop_sample.index[0])
print(drop_out)

print(' -) Remove a row observation. This will remove second and third row')
drop_out = Y_drop_sample.drop([1,2])
print(drop_out)

print(' -) Remove last n=-3 rows')
drop_out = Y_drop_sample[:-3]
print(drop_out)

print(' -) Remove others rows, take only n=1 row')
drop_out = Y_drop_sample[:1]
print(drop_out)

print('** Pandas: Find examples on dataset **')
print(' -) Find indexes of DataFrame that contains numbers')
indexNames = Y_drop_sample[ Y_drop_sample['sepal_length'] == 5.1 ].index
print(indexNames)

indexNames = Y_drop_sample[ Y_drop_sample['petal_length'] == 1.4 ].index
print(indexNames)

indexNames = Y_drop_sample[ (Y_drop_sample['sepal_width'] >= 3.0) & (Y_drop_sample['sepal_width'] <= 3.3) ].index
print(indexNames)

print('** Pandas: Axis examples **')
'''
IND    a    b 
0      10   22  = 32
1      30   44  = 74
2      60   55  = 115
3      80   77  = 157
4      90  101  = 191
_________________
      270  299
'''
srs_a = pd.Series([10,30,60,80,90])
srs_b = pd.Series([22, 44, 55, 77, 101])
df = pd.DataFrame({'a': srs_a, 'b': srs_b})
print(' -) Sum on axis=0. We want operate on columns')
print(df.sum(axis=0))
print(' -) Sum on axis=1. We want to operate on rows')
print(df.sum(axis=1))

print('** Pandas: Find examples on dataset **')

print('** Pandas: LOC and ILOC **')
#Copy a dataset
Y_drop_sample_copy = Y_drop_sample.copy()
# Shuffle
Y_drop_sample_copy = Y_drop_sample.sample(frac=1)
print(Y_drop_sample_copy.head())

print('-) ILOC **')
print(Y_drop_sample_copy.iloc[3])
print(Y_drop_sample_copy.iloc[3])

print('-) LOC **')
print(Y_drop_sample_copy.loc[0])
print('-) LOC: value on column name')
print(Y_drop_sample_copy.loc[0,"species"])


