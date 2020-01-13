import pandas as pd
import numpy as np

ds_filename = "pandas_dataset_examples/housing.data"
'''
Example 1 - Simple Linear regression on two variables.
'''
## dataset loading
pd_dataset = pd.read_csv(ds_filename, sep="\s+", usecols=[5,13], names=["RM", "MEDV"])
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

# Dataset division train/test
# Dataset division using sklearn
from sklearn.model_selection import train_test_split
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

## applay the LinearRegression on dataset witk sklearn
print('*** LinearRegression with sklearn ***')
from sklearn.linear_model import LinearRegression
ll = LinearRegression()
out = ll.fit(X_train,Y_train)
Y_pred = ll.predict(X_test)
print('*** Predicted value output ***')
print(Y_pred)

print('*** Evaluate LinearRegression with sklearn ***')
'''
The sklearn library evaluate a model using the Mean Squared Error, MSE

'''
print('*** Math Approach ***')
RSS = ( (Y_test - Y_pred)**2).sum()
print('RSS: ' + str(RSS))

MSE = np.mean( (Y_test - Y_pred)**2 )
print('MSE: ' + str(MSE))

RMSE = np.sqrt(MSE)
print('RMSE: ' + str(RMSE))

print('*** sklearn Library Approach ***')
from sklearn.metrics import mean_squared_error

MSE_SL_LIB = mean_squared_error(Y_test,Y_pred)
print('MSE_SL_LIB: ' + str(MSE_SL_LIB))

print('*** Scoring function: determination coefficient => R Square index ***')
# Return a score from 0-1 and will be negative:
# close to 1: better quality of model
from sklearn.metrics import r2_score
print('R square index: R^2')
r2_score_out = r2_score(Y_test,Y_pred)
print('r^2_SL_LIB: ' + str(r2_score_out))
