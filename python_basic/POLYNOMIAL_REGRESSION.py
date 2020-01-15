import pandas as pd
import numpy as np
import matplotlib as mtplt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# IMPO: add tabulate to pretty print python datas
from tabulate import tabulate

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

cols = ["RM","LSTAT","DIS","RAD","MEDV"]
sns.pairplot(pd_dataset[cols])
plt.show()

# Create numpy arrays
X = pd_dataset[["LSTAT"]].values
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

## POLYNOMIAL ANALYSIS ##
from sklearn.preprocessing import PolynomialFeatures
polynomial_degree = 2
polyfeats = PolynomialFeatures(degree=polynomial_degree)

# Build the polinomyal feature
X_train_poly = polyfeats.fit_transform(X_train)
X_test_poly = polyfeats.fit_transform(X_test)

print('*** Compare train sets ***')
print('-) Normal train set')
print(X_train[:5])
print('-) Polynomial train sets')
'''
IMPO: using the polinomyal approach, make the set composet of 
degrees+1 columns.

If degree was 2, we have 3 columns
-) first column is the feature raised to 0 (so it gives always 1 as result)
-) second column is the feature raised to 1 (so it gives the number itself)
-) third column is the feature raised to 2 (the degree we have specified)
'''
print(X_train_poly[:5])

# Polynomial regression: degree comparison
## Using a subset of features
for i in range(1,11):
    polyfeats = PolynomialFeatures(degree=i)
    X_train_poly = polyfeats.fit_transform(X_train)
    X_test_poly = polyfeats.fit_transform(X_test)
    #
    ll = LinearRegression()
    ll.fit(X_train_poly,Y_train)
    Y_pred = ll.predict(X_test_poly)
    #
    print(' *************** ')
    print('Degree: ' + str(i))
    print('Prediction output sample')
    print(Y_pred[:5])
    #
    MSE = mean_squared_error(Y_test, Y_pred)
    r2_score_out = r2_score(Y_test, Y_pred)
    print("MSE: " + str(MSE))
    print("R2 score: " + str(r2_score_out))
    # Plot LinearRegression Coefficient
    weigth_of_linearRegression = ll.coef_
    print('Weigth of calculated linearRegression: ' + str(weigth_of_linearRegression))
    bias_or_intercept_of_linearRegression = ll.intercept_
    print('Bias of calculated linearRegression: ' + str(bias_or_intercept_of_linearRegression))
    print(' *************** ')

## Using all of features dataset
print(' ** ANALISY OF POLINOMYAL REGRESSION USING ALL DATASET FEATURES ** ')
X = pd_dataset.drop("MEDV", axis=1).values
Y = pd_dataset["MEDV"].values

# Create set for train/test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=0)

print('Analyzing from 1 to 4 degree')
## Remeber it's numbered from N+1
max_degree = 5

for i in range(1,5):
    polyfeats = PolynomialFeatures(degree=i)
    X_train_poly = polyfeats.fit_transform(X_train)
    X_test_poly = polyfeats.fit_transform(X_test)
    #
    ll = LinearRegression()
    ll.fit(X_train_poly,Y_train)
    Y_pred = ll.predict(X_test_poly)
    #
    print(' *************** ')
    print('Degree: ' + str(i))
    print('Prediction output sample')
    print(Y_pred[:5])
    #
    MSE = mean_squared_error(Y_test, Y_pred)
    r2_score_out = r2_score(Y_test, Y_pred)
    print("MSE: " + str(MSE))
    print("R2 score: " + str(r2_score_out))
    # Plot LinearRegression Coefficient
    weigth_of_linearRegression = ll.coef_
    print('Weigth of calculated linearRegression: ' + str(weigth_of_linearRegression))
    bias_or_intercept_of_linearRegression = ll.intercept_
    print('Bias of calculated linearRegression: ' + str(bias_or_intercept_of_linearRegression))
    print(' *************** ')