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

ds_filename = "pandas_dataset_examples/iris.csv"
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
#Get data as numpy array
target_property="species"
data_as_numpy_array = pd_dataset.drop(target_property, axis=1).values
target_as_numpy_array = pd_dataset[target_property].values

print('** Numpy array printing **')
print('-) Data')
print(data_as_numpy_array[:5])
print('-) Target classes')
print(target_as_numpy_array[:5])

X_train, X_test, Y_train, Y_test = train_test_split(data_as_numpy_array,target_as_numpy_array,test_size=0.3)
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

#
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)
#
ss =  StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
##
X2_train = X_train[:,:2]
X2_test = X_test[:,:2]

## Build SVM Model
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X2_train,Y_train)

# Metrics calucaltion for SVC
#1. Accuracy
print('-) Calculate the performance of models TRAIN DS')
accuracy_train = svc.score(X2_train, Y_train)
print('Accuracy (train): %.4f' % accuracy_train)
accuracy_test = svc.score(X2_test, Y_test)
print('Accuracy (test): %.4f' % accuracy_test)