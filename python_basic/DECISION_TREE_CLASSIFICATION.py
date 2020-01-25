#
import collections

import pandas as pd

import numpy as np

#
import matplotlib as mtplt
import matplotlib.pyplot as plt

# seaborn
import seaborn as sns

# sklearn
## sklearn.preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

## sklearn.metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

## sklearn.model_selection
from sklearn.model_selection import train_test_split

## sklearn.linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

'''
IMPO This example uses the titanic dataset.
'''

ds_filename = "pandas_dataset_examples/titanic.csv"

## dataset loading
pd_dataset = pd.read_csv(ds_filename)
print('*) DS Info')
print(pd_dataset.info())
print('-) DS shape')
print(pd_dataset.shape)
print('-) DS  describe()')
print(pd_dataset.describe())
print('-) DS sample data')
print(pd_dataset.head())

#
target_variable = "Survived"
#Remove Name
feature_to_remove_1 = "Name"
pd_dataset = pd_dataset.drop(feature_to_remove_1,axis=1)
# Transform categorical value like Sex
pd_dataset = pd.get_dummies(pd_dataset)
print('-) DS sample data')
print(pd_dataset.head())

# Crate numpy array
X = pd_dataset.drop(target_variable, axis=1)
Y = pd_dataset[target_variable].values

#
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)
print('DS subdivision: ')
print('*** X_train ***')
print('X_train shape: ' + str(X_train.shape) )
print('X_train sample data ')
print(X_train[:5])
#
print('*** X_test ***')
print('X_test shape: ' + str(X_test.shape) )
print('X_test sample data ')
print(X_test[:5])
#
print('*** Y_train ***')
print('Y_train shape: ' + str(Y_train.shape) )
print('Y_train sample data ')
print(Y_train[:5])
#
print('*** Y_test ***')
print('Y_test shape: ' + str(Y_test.shape) )
print('Y_test sample data ')
print(Y_test[:5])

# IMPO: The descision tree not require that data are on same scale,
# so no normalization/standardization approach need to be executed.

# Create model for Decision tree
from sklearn.tree import DecisionTreeClassifier
dtModel = DecisionTreeClassifier(criterion="gini")
dtModel.fit(X_train,Y_train)

Y_pred_train = dtModel.predict(X_train)
Y_pred_test = dtModel.predict(X_test)

print('-) Calculate the performance of models TRAIN DS')
accuracy_train = accuracy_score(Y_train, Y_pred_train)
print('Accuracy (train): %.4f' % accuracy_train)
accuracy_test = accuracy_score(Y_test, Y_pred_test)
print('Accuracy (test): %.4f' % accuracy_test)

# Approach to optimization:
# 1. Reduce the deep of tree
print('** Example 1: Reduce the deep of a tree **')
dtModel = DecisionTreeClassifier(criterion="gini", max_depth=6)
dtModel.fit(X_train,Y_train)

Y_pred_train = dtModel.predict(X_train)
Y_pred_test = dtModel.predict(X_test)

print('-) Calculate the performance of models TRAIN DS')
accuracy_train = accuracy_score(Y_train, Y_pred_train)
print('Accuracy (train): %.4f' % accuracy_train)
accuracy_test = accuracy_score(Y_test, Y_pred_test)
print('Accuracy (test): %.4f' % accuracy_test)

# Show the graphs of decision tree
from sklearn.tree import export_graphviz

dotfile = open("tree.dot","w")
export_graphviz(dtModel, out_file=dotfile, feature_names= pd_dataset.columns.drop(target_variable))
dotfile.close()
'''
To visualize the generated graph:
http://www.webgraphviz.com/
'''
##

