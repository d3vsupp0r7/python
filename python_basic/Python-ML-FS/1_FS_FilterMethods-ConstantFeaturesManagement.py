import pandas as pd
import numpy as np

#
#data = pd.read_csv('C:/dev/nlp_dataset/santander-customers-satisfaction-train.csv/train.csv')
#print(data.shape)

#####
ds_filename_train = "dataset/stand-cust-sat-train.csv"
ds_filename_test = "dataset/stand-cust-sat-test.csv"

pd_train_dataset = pd.read_csv(ds_filename_train)
pd_test_dataset  = pd.read_csv(ds_filename_test)

print("*) Train dataset info")
print(pd_train_dataset.shape)

print("*) Train dataset head()")
print(pd_train_dataset.head())

print("*) Train dataset tail()")
print(pd_train_dataset.tail())

print("*) Train dataset describe()")
print(pd_train_dataset.describe())

pd_train_dataset_reduced = pd.read_csv(ds_filename_train, nrows=50000)
print("*) Train dataset reduced info")
print(pd_train_dataset_reduced.shape)

print("*) Train dataset reduced head()")
print(pd_train_dataset_reduced.head())

print("*) Train dataset reduced tail()")
print(pd_train_dataset_reduced.tail())

print("*) Train dataset reduced describe()")
print(pd_train_dataset_reduced.describe())

#
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

[col for col in pd_train_dataset_reduced.columns if pd_train_dataset_reduced[col].isnull().sum() > 0]
"# separate dataset into train and test"
#X_train, X_test, y_train, y_test = train_test_split(
#    data.drop(labels=['TARGET'], axis=1),
#    data['TARGET'],
#    test_size=0.3,
#    random_state=0)

X_train, X_test, y_train, y_test = train_test_split(
pd_train_dataset_reduced.drop(labels=['TARGET'], axis=1),
pd_train_dataset_reduced['TARGET'],
test_size=0.3,
random_state=0)

print(X_train.shape)
print(X_test.shape)

### Using variance threshold from sklearn
sel = VarianceThreshold(threshold=0)
sel.fit(X_train)  # fit finds the features with zero variance

# get_support is a boolean vector that indicates which features are retained
# if we sum over get_support, we get the number of features that are not constant
print(sum(sel.get_support()))

# another way of finding non-constant features is like this
print(len(X_train.columns[sel.get_support()]))

# finally we can print the constant features
print(len([x for x in X_train.columns
 if x not in X_train.columns[sel.get_support()]
 ]))

[x for x in X_train.columns if x not in X_train.columns[sel.get_support()]]

print(X_train.columns)
print(type(X_train.columns))

constant_columns = [column for column in X_train.columns
                    if column not in X_train.columns[sel.get_support()]]
print(len(constant_columns))
for column in constant_columns:
    print(column)

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