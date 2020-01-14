import pandas as pd
import numpy as np
import matplotlib as mtplt
import matplotlib.pyplot as plt

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
data_as_numpy_array = pd_dataset.drop("MEDV", axis=1).values
target_as_numpy_array = pd_dataset["MEDV"].values

#
print('Get information about properties on dataset')
print(pd_dataset.info())

# Analize wat type of feature we can select to execute preliminary analysis
# using the correlation index
print('Correlation index analysis')
print(pd_dataset.corr())

# IMPO: add tabulate to pretty print python datas
from tabulate import tabulate
#print(tabulate(pd_dataset, headers='keys', tablefmt='psql'))
print(tabulate(pd_dataset.corr(), headers='keys', tablefmt='psql'))

# seaborn polt library
import seaborn as sns
print('The seaborn version is {}.'.format(sns.__version__))
print('The matplotlib version is {}.'.format(mtplt.__version__))

# Creating heatmap using seaborn
data_to_plot = sns.heatmap(pd_dataset.corr(), xticklabels=pd_dataset.columns, yticklabels=pd_dataset.columns)
plt.show()

# Creating annotated heatmap using seaborn
# data_to_plot = sns.heatmap(pd_dataset.corr(), xticklabels=pd_dataset.columns, yticklabels=pd_dataset.columns, annot=True)
# plt.show()

'''
Notes:
This script was executed with following configuration:
-) The seaborn version is 0.9.0.
-) The matplotlib version is 3.1.1
But you may expect that the output of seaborn heatmap was no drawn correctly.

For workaround, update matplotlib to 3.1.2 or downgrade to 3.0.3

'''
cols = ["RM","LSTAT","PTRATIO","TAX","INDUS","MEDV"]
plt.figure(figsize=(15,15))
data_to_plot = sns.heatmap(pd_dataset[cols].corr(),
                           xticklabels=pd_dataset[cols].columns,
                           yticklabels=pd_dataset[cols].columns,
                           annot=True,
                           annot_kws={'size':12})

plt.show()

## Using seaborn to create pair graphs
'''
Using a pariplot, we will consider only plots that have our target variable on Y axis.
In the boston hous example, MEDV is our Y (target variable).
'''
sns.pairplot(pd_dataset[cols])
plt.show()

'''
Using RM an LSTAT for model creation.
RM-MEDV correlation    :  0.69 (Positive significat correlation)
LSTAT-MEDV correlation : -0.73 (Inverse correlation)
'''
X = pd_dataset[["RM","LSTAT"]].values
Y = pd_dataset["MEDV"].values

# Train model
from sklearn.model_selection import train_test_split
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

## Apply the LinearRegression on dataset witk sklearn
print('*** LinearRegression with sklearn ***')
from sklearn.linear_model import LinearRegression
ll = LinearRegression()
out = ll.fit(X_train,Y_train)
Y_pred = ll.predict(X_test)
print('*** Predicted value output ***')
print(Y_pred)

print('*** Model metrics ***')
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

MSE = mean_squared_error(Y_test,Y_pred)
r2_score_out = r2_score(Y_test,Y_pred)
print("MSE: " + str(MSE))
print("R2 score: " + str(r2_score_out))

# Plot LinearRegression Coefficient
weigth_of_linearRegression =  ll.coef_
print('Weigth of calculated linearRegression: ' + str(weigth_of_linearRegression))
bias_or_intercept_of_linearRegression = ll.intercept_
print('Bias of calculated linearRegression: ' + str(bias_or_intercept_of_linearRegression))

# Make a single prediction
print('Example: Test a Linear Regression model on single input ')
print('Prediction Example 1')
value_to_predict = [6.0300, 7.88]
y_single_pred = ll.predict([ [6.0300, 7.88] ])
print('Predicted value for input: ' + str(value_to_predict) + ' is: ' + str(y_single_pred))
print('Prediction Example 2')
value_to_predict = [6.7940, 6.48]
y_single_pred = ll.predict([ [6.7940, 6.48] ])
print('Predicted value for input: ' + str(value_to_predict) + ' is: ' + str(y_single_pred))
print('Prediction Example 3')
value_to_predict = [5.349, 19.77]
y_single_pred = ll.predict([ [5.349, 19.77] ])
print('Predicted value for input: ' + str(value_to_predict) + ' is: ' + str(y_single_pred))

## **************************************** ##
# analisys of dataset with all properties to see if there are some differences
# step1:
X = pd_dataset.drop("MEDV", axis=1).values
Y = pd_dataset["MEDV"].values

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

# Step 2: normalize Dataset
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)

# Execute the linear regression
ll = LinearRegression()
ll.fit(X_train_std, Y_train)
Y_pred_std = ll.predict(X_test_std)

MSE_with_std = mean_squared_error(Y_test,Y_pred_std)
r2_score_out_std = r2_score(Y_test,Y_pred_std)
print("MSE with std: " + str(MSE_with_std))
print("R2 score std: " + str(r2_score_out_std))

# Plot LinearRegression Coefficient
weigth_of_linearRegression_std =  ll.coef_
print('Weigth of calculated linearRegression std: ' + str(weigth_of_linearRegression_std))
bias_or_intercept_of_linearRegression_std = ll.intercept_
print('Bias of calculated linearRegression std: ' + str(bias_or_intercept_of_linearRegression_std))

# Print data in pritty form
print('Print coefficient matrix with names')
print(list(zip(pd_dataset.columns, ll.coef_)) )
#print(zip(pd_dataset.columns, ll.coef_))
#out_print = list(zip(pd_dataset.columns, ll.coef_))

#for i in range(len(out_print)):
#    for x in out_print:
#        print(x[i], end =' ')
#    print()

print(" *********************************************** ")
print(" **                ANALYSIS   RESULTS         ** ")
print(" *********************************************** ")
print(" **         LINEAR REGRESSION EQUATION        ** ")
print(" **-------------------------------------------** ")
print(" **                COEFFICIENTS               ** ")
print(" **                BIAS                       ** ")
print(" *********************************************** ")
print(" **               MATH INDICATORS             ** ")
print(" **-------------------------------------------** ")
print(" **               MSE (Mean Square Error)     ** ")
print(" **               R Square                    ** ")
print(" *********************************************** ")