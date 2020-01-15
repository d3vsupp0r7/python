import pandas as pd
import numpy as np
import matplotlib as mtplt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

'''
This example was executed on boston housing dataset.

'''

ds_filename = "pandas_dataset_examples/housing.data"

## dataset loading
pd_dataset = pd.read_csv(ds_filename, sep="\s+"
                         , names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"])
print('*) DS Info')
print('-) DS shape')
print(pd_dataset.shape)
print('-) DS  desc')
print(pd_dataset.describe())
print('-) DS sample data')
print(pd_dataset.head())

#Get data as numpy array
X = pd_dataset.drop("MEDV", axis=1).values
Y = pd_dataset["MEDV"].values

# Create set for train/test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=0)

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

## 1. Generate overfitting
poly_feats = PolynomialFeatures(degree=2)
X_train_poly = poly_feats.fit_transform(X_train)
X_test_poly = poly_feats.transform(X_test)
print('DS Train shape')
print(X_train_poly.shape)
# 2. standardization of sataset
ss = StandardScaler()
X_train_poly = ss.fit_transform(X_train_poly)
X_test_poly = ss.fit_transform(X_test_poly)
#
ll = LinearRegression()
ll.fit(X_train_poly, Y_train)
#
Y_pred_train = ll.predict(X_train_poly)
mse_train = mean_squared_error(Y_train, Y_pred_train)
r2_train = r2_score(Y_train, Y_pred_train)
print('DS Train information')
print('MSE train: ' + str(mse_train) )
print('R square train: ' + str(r2_train) )
#
Y_pred_test = ll.predict(X_test_poly)
mse_test = mean_squared_error(Y_test, Y_pred_test)
r2_test = r2_score(Y_test, Y_pred_test)
print('DS Test information')
print('MSE test: ' + str(mse_test) )
print('R square test: ' + str(r2_test) )
