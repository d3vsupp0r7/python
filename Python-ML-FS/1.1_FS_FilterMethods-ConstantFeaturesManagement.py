import pandas as pd
import numpy as np
'''
Feature Selection: Filter Methods - Manage Constant Features

Constant features are the type of features that contain only one value for all the observations (rows) in the dataset.

'''
#####
#1. Datset loading
ds_filename_train = "dataset/stand-cust-sat-train.csv"
ds_filename_test = "dataset/stand-cust-sat-test.csv"

pd_train_dataset = pd.read_csv(ds_filename_train)
pd_test_dataset  = pd.read_csv(ds_filename_test)

print("***************************")
print("** TRAIN DATASET ANALYSIS ")
print("***************************")

print("** TRAIN DATASET ANALYSIS: INTIAL INFO")
print("*) Train dataset info")
print(pd_train_dataset.shape)

print("*) Train dataset head()")
print(pd_train_dataset.head())

print("*) Train dataset tail()")
print(pd_train_dataset.tail())

print("*) Train dataset describe()")
print(pd_train_dataset.describe())

print("** TRAIN REDUCED DATASET ANALYSIS ")
#1. Load a subset rows of dataset
'''
In all feature selection procedures, it is good practice to select the features by examining only the training set. 
And this is to avoid overfitting.
'''
pd_train_dataset_reduced = pd.read_csv(ds_filename_train, nrows=50000)
print("*) Train dataset reduced info")
print(pd_train_dataset_reduced.shape)

print("*) Train dataset reduced head()")
print(pd_train_dataset_reduced.head())

print("*) Train dataset reduced tail()")
print(pd_train_dataset_reduced.tail())

print("*) Train dataset reduced describe()")
print(pd_train_dataset_reduced.describe())

# Use of SKLEANR library to identify constant features
print("## IDENTIFY CONSTANT FEATURES")
print("## Identify constant features of dataset using SKLEARN library")
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

# 1. check the presence of null data.
print("*) Checks if there are null data")
null_data = [col for col in pd_train_dataset_reduced.columns if pd_train_dataset_reduced[col].isnull().sum() > 0]
print(len(null_data))
for column in null_data:
    print(column)

#2. separate dataset into train and test
print("*) Slit dataset data into test and train dataset")
X_train, X_test, y_train, y_test = train_test_split(
pd_train_dataset_reduced.drop(labels=['TARGET'], axis=1),
pd_train_dataset_reduced['TARGET'],
test_size=0.3,
random_state=0)

print("\t*) TRAIN DATASET DIMENSION")
print(X_train.shape)
print("\t*) TEST DATASET DIMENSION")
print(X_test.shape)

### 3. Using variance threshold from sklearn
'''
Variance threshold from sklearn is a simple baseline approach to feature selection. 
It removes all features which variance doesn’t meet some threshold. By default, it removes all 
zero-variance features, i.e., features that have the same value in all samples.
'''
sel = VarianceThreshold(threshold=0)
sel.fit(X_train)  # fit finds the features with zero variance

# get_support is a boolean vector that indicates which features are retained
# if we sum over get_support, we get the number of features that are not constant
print("*) Number of feature that are NOT CONSTANT using get_support() on VarianceThreshold")
print(sum(sel.get_support()))

# another way of finding non-constant features is like this
print("*) Number of feature that are NOT CONSTANT using get_support() on train columns dataset")
print(len(X_train.columns[sel.get_support()]))

# finally we can print the constant features
print("*) Print names of features that are CONSTANT using get_support() on VarianceThreshold")
print(len([x for x in X_train.columns
 if x not in X_train.columns[sel.get_support()]
 ]))

[x for x in X_train.columns if x not in X_train.columns[sel.get_support()]]

print("*) Columns of train dataset")
print(X_train.columns)
print("*) Columns type of train dataset")
print(type(X_train.columns))
'''
We can see that 58 columns / variables are constant. This means that 58 variables show the same value, just one value,
for all the observations of the training set.
'''
constant_columns = [column for column in X_train.columns
                    if column not in X_train.columns[sel.get_support()]]
print("*) Number of constant features")
print(len(constant_columns))
print("*) Names of constant features")
for column in constant_columns:
    print(column)

# let's visualise the values of one of the constant variables
# as an example
constant_feature_name = "ind_var2_0"
print("Example of constant_feature: " + constant_feature_name)
print(X_train[constant_feature_name].unique())
'''
to remove constant features from training and test sets, we can use the transform() method of the sel. 

'''
print("## Remove constant features from dataset ")
print("\t ORIGINAL dataset ")
print("\t*) TRAIN DATASET DIMENSION")
print(X_train.shape)
print("\t*) TEST DATASET DIMENSION")
print(X_test.shape)
#
print("\t REMOVED CONSTANT FEATURES FROM dataset ")
X_train = sel.transform(X_train)
X_test = sel.transform(X_test)
print("\t*) TRAIN DATASET DIMENSION")
print(X_train.shape)
print("\t*) TEST DATASET DIMENSION")
print(X_test.shape)
'''
execute the above script, you will see that both our training and test sets will now contain 320 columns, 
since the 50 constant columns have been removed.
'''
print("***************************")
print("** TRAIN DATASET - CONSTANT FEATURE REMOVAL - ANOTHER APPROACH")
print("***************************")
pd_train_dataset_reduced = pd.read_csv(ds_filename_train, nrows=50000)
#2. separate dataset into train and test
print("*) Split dataset data into test and train dataset")
X_train, X_test, y_train, y_test = train_test_split(
pd_train_dataset_reduced.drop(labels=['TARGET'], axis=1),
pd_train_dataset_reduced['TARGET'],
test_size=0.3,
random_state=0)

print("\t*) TRAIN DATASET DIMENSION")
print(X_train.shape)
print("\t*) TEST DATASET DIMENSION")
print(X_test.shape)

print("\t*) Number of constant feature into train dataset")
constant_features = [
    feat for feat in X_train.columns if X_train[feat].std() == 0]
print(len(constant_features))

print("\t*) Dropping constant features from TRAIN DATASET")
# we can then drop these columns from the train and test sets
X_train.drop(labels=constant_features, axis=1, inplace=True)
X_test.drop(labels=constant_features, axis=1, inplace=True)
print("\t*) TRAIN DATASET DIMENSION")
print(X_train.shape)
print("\t*) TEST DATASET DIMENSION")
print(X_test.shape)

print("***************************")
print("## MANAGING CATEGORICAL VARIABLES")
pd_train_dataset_categorical = pd.read_csv(ds_filename_train, nrows=50000)

#2. separate dataset into train and test
print("*) Split dataset data into test and train dataset")
X_train, X_test, y_train, y_test = train_test_split(
pd_train_dataset_reduced.drop(labels=['TARGET'], axis=1),
pd_train_dataset_reduced['TARGET'],
test_size=0.3,
random_state=0)

print("\t*) TRAIN DATASET DIMENSION")
print(X_train.shape)
print("\t*) TEST DATASET DIMENSION")
print(X_test.shape)

print("*) Execute a simple categorical transformation of variable for example purpose")
X_train = X_train.astype('O')
print(X_train.dtypes)

# and now find those columns that contain only 1 label
constant_features = [
 feat for feat in X_train.columns if len(X_train[feat].unique()) == 1]
# Same as before, we observe 58 variables that show only 1 value across all the observations of the dataset. We can
# appreciate the usefulness of looking out for constant variables at the beginning of any modeling exercise.
print(len(constant_features))

#from tabulate import tabulate
#print(tabulate(constant_columns))


'''
def variance_threshold_selector(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

print(variance_threshold_selector(X_train, 0) )
'''


###########
print("***************************")
print("** TEST DATASET")
print("***************************")
print("*) Test dataset info")
print(pd_test_dataset.shape)