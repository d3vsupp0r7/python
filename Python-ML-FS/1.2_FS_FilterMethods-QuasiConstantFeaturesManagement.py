import pandas as pd
import numpy as np
'''
Feature Selection: Filter Methods - Manage Quasi-Constant Features
Quasi-constant features  have the same values for a very large subset of the observations. 
Such features are not very useful for making predictions. 
There is no rule as to what should be the threshold for the variance of quasi-constant features. 
However, as a suggested practice is to remove those quasi-constant features that have more than 99% similar 
values for the output observations.

To proceed into management of quasiconstant feature we need firsto of all remove the constant feature.
from this result after we proceeed to manage the qausi-constant features.
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

# Use of SKLEANR library to identify quasi-constant features
print("## IDENTIFY QUASI-CONSTANT FEATURES")
print("## Identify quasi-constant features of dataset using SKLEARN library")
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

# 1. check the presence of null data.
print("*) Checks if there are null data")
null_data = [col for col in pd_train_dataset_reduced.columns if pd_train_dataset_reduced[col].isnull().sum() > 0]
print(len(null_data))
for column in null_data:
    print(column)

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
#########

#########
'''
execute the above script, you will see that both our training and test sets will now contain 320 columns, 
since the 50 constant columns have been removed.
'''
# 0.1 indicates 99% of observations approximately
pd_train_dataset_reduced = pd.read_csv(ds_filename_train, nrows=50000)
print("*) Split dataset data into test and train dataset")
X_train, X_test, y_train, y_test = train_test_split(
pd_train_dataset_reduced.drop(labels=['TARGET'], axis=1),
pd_train_dataset_reduced['TARGET'],
test_size=0.3,
random_state=0)

sel_2 = VarianceThreshold(threshold=0.01)
sel_2.fit(X_train)  # fit finds the features with low variance
print("*) Number of feature that are NOT QUASI-CONSTANT using get_support() on VarianceThreshold")
print(sum(sel_2.get_support()))
constant_columns = [column for column in X_train.columns
                    if column not in X_train.columns[sel_2.get_support()]]
print("*) Names of quasi-constant features")
for column in constant_columns:
    print(column)

#Sort
print("** SORT **")
constant_columns.sort()
print(constant_columns)
for column in constant_columns:
    print(column)
#Sorting string with numbers inside
import re
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]
constant_columns.sort(key=natural_keys)
print("** SORT WITH STRING AND NUMBERS **")
for column in constant_columns:
    print(column)
#############
print('## Info for variable')
print(X_train['ind_var31'].value_counts()/np.float(len(X_train)))
###
X_train = sel_2.transform(X_train)
X_test = sel_2.transform(X_test)
print("\t*) TRAIN DATASET DIMENSION")
print(X_train.shape)
print("\t*) TEST DATASET DIMENSION")
print(X_test.shape)

print("***************************")
print("** TRAIN DATASET - QUASI-CONSTANT FEATURE REMOVAL - ANOTHER APPROACH")
print("***************************")
print("[1] Load data")
pd_train_dataset_reduced = pd.read_csv(ds_filename_train, nrows=50000)
print("[2] Split dataset data into test and train dataset")
X_train, X_test, y_train, y_test = train_test_split(
pd_train_dataset_reduced.drop(labels=['TARGET'], axis=1),
pd_train_dataset_reduced['TARGET'],
test_size=0.3,
random_state=0)

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