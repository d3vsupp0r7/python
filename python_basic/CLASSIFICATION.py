#
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
from sklearn.preprocessing import LabelEncoder

ds_filename = "pandas_dataset_examples/wdbc.data"

pd_dataset = pd.read_csv(ds_filename,names=["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean",
                                            "smoothness_mean","compactness_mean","concavity_mean","concave points_mean",
                                            "symmetry_mean","fractal_dimension_mean","radius_se","texture_se",
                                            "perimeter_se","area_se","smoothness_se","compactness_se","concavity_se",
                                            "concave points_se","symmetry_se","fractal_dimension_se","radius_worst",
                                            "texture_worst","perimeter_worst","area_worst","smoothness_worst",
                                            "compactness_worst","concavity_worst","concave points_worst",
                                            "symmetry_worst","fractal_dimension_worst"])

print('*) DS Info')
print(pd_dataset.info())
print('-) DS shape')
print(pd_dataset.shape)
print('-) DS describe()')
print(pd_dataset.describe())
print('-) DS sample data')
print(pd_dataset.head())
print('-) DS correlation information')
from tabulate import tabulate
print(tabulate(pd_dataset.corr()))

print('* 1. Determinate list of classes to predict')
class_column = "diagnosis"
print(pd_dataset[class_column].unique() )

##
feature_selection_1 = "radius_se"
feature_selection_2 = "concave points_worst"
X = pd_dataset[[feature_selection_1,feature_selection_2]].values
Y = pd_dataset[class_column].values

print('-) DS: feature selection sample data')
print(X[:5])

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

## -- Transform numerical-- ##
'''
IMPO 
'''
print('*** Encode categorical data ***')
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)
#
print('Print encoded train set')
print(Y_train[:5])
#
print('*** Dataset  ***')