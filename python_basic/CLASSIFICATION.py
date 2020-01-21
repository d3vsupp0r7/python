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
print('*** Dataset Standardization  ***')
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Build LogisticRegression model => Classification model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
out = lr.fit(X_train,Y_train)
print(out)

# Model testing
Y_pred = lr.predict(X_test)
print('Predict')
print(Y_pred[:5])
# Whtai is the probability
print('Predict: proba')
Y_pred_proba = lr.predict_proba(X_test)
print(Y_pred_proba[:5])
print('Predict: Y_pred_log_proba')
Y_pred_log_proba = lr.predict_log_proba(X_test)
print(Y_pred_log_proba[:5])

# Accuracy test
'''
Accuracy require the exact prediction as parameter.
The accuracy variate from range 0->1
0 near value means wrong prediction
1 near values means exact prediction
'''
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred)
print('Accuracy: ' + str(accuracy))
# Log likelyhood
'''
IMPO: Log Loss
The sklearn library implement the log loss likelyhood,
this means that it's implement the nagative of log likelihood.
This library require, for it's prediction the probability.
The output of this function variate from 0 to 1
Values close to 0 means high precision and values near to 1 means less precision.
'''
from sklearn.metrics import log_loss
logLoss = log_loss(Y_test, Y_pred_proba)
print('Log Loss: ' + str(logLoss))

# Print the decision boundary
def show_bounds(model,X,Y,labels=["Class 0","Class 1"], figsize=(12,10), graphTitle="Graph title"):

    fig = plt.figure(figsize=figsize)

    h = .02

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    X_m = X[Y==1]
    X_b = X[Y==0]
    plt.scatter(X_b[:, 0], X_b[:, 1], c="green",  edgecolor='white', label=labels[0])
    plt.scatter(X_m[:, 0], X_m[:, 1], c="red",  edgecolor='white', label=labels[1])
    plt.legend()
    fig.suptitle(graphTitle, fontsize=20)
    plt.show()

show_bounds(lr,X_train,Y_train,labels=["B","M"],graphTitle="Train decision boundary")

show_bounds(lr,X_test,Y_test,labels=["B","M"],graphTitle="Test decision boundary")

## Refactoring with dataset manipulation: all features without 'id' feature ##
print('***** Refactoring of dataset: Remove id column *****')
drop_feature_1 = "id"
X = pd_dataset.drop([class_column,drop_feature_1],axis=1).values
Y = pd_dataset[class_column].values
print('Feature: DS sample data')
print(X[:5])
print('Label/Target: DS sample data')
print(Y[:5])

#
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
print('*** Dataset Standardization  ***')
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

## Apply the LogisticRegression model to manipulated dataset ##
lr = LogisticRegression()
out = lr.fit(X_train,Y_train)
print(out)

# Model testing
Y_pred = lr.predict(X_test)
print('Predict')
print(Y_pred[:5])
# Whtai is the probability
print('Predict: proba')
Y_pred_proba = lr.predict_proba(X_test)
print(Y_pred_proba[:5])
print('Predict: Y_pred_log_proba')
Y_pred_log_proba = lr.predict_log_proba(X_test)
print(Y_pred_log_proba[:5])

# Print model metrics
accuracy_ds_2 = accuracy_score(Y_test, Y_pred)
print('[] Accuracy ds 2: ' + str(accuracy_ds_2))
logLoss_ds_2 = log_loss(Y_test, Y_pred_proba)
print('[] Log Loss ds 2: ' + str(logLoss_ds_2))

# Comparison metrics with all features
print('[] Accuracy ds 1: ' + str(accuracy))
print('[] Log Loss ds 1: ' + str(logLoss))

# Regularization of LogisticRegression
'''
The LogisticRegression implements the regularization with two parameters:
penalty => assumes the l1/l2 values for regularization type we want to use
C       => is the inverse of lambda value (1/lambda). If this value is high makes the regulation more weak
                                                    otherwise make the regulation more strong
By default these values are:
penalty = l2
C = 1
'''
## Apply the LogisticRegression model to manipulated dataset
##   with l1 regularization
lr = LogisticRegression(penalty="l1", C=1)
out = lr.fit(X_train,Y_train)
print(out)

# Model testing
Y_pred = lr.predict(X_test)
print('Predict')
print(Y_pred[:5])
# Whtai is the probability
print('Predict: proba')
Y_pred_proba = lr.predict_proba(X_test)
print(Y_pred_proba[:5])
print('Predict: Y_pred_log_proba')
Y_pred_log_proba = lr.predict_log_proba(X_test)
print(Y_pred_log_proba[:5])

# Print model metrics
accuracy_ds_reg_l1 = accuracy_score(Y_test, Y_pred)
print('[] Accuracy ds reg l1: ' + str(accuracy_ds_reg_l1))
logLoss_ds_reg_l1 = log_loss(Y_test, Y_pred_proba)
print('[] Log Loss ds reg l1: ' + str(logLoss_ds_reg_l1))

# Comparison metrics with all features
print('[] Accuracy ds 1: ' + str(accuracy))
print('[] Log Loss ds 1: ' + str(logLoss))
print('[] Accuracy ds 2: ' + str(accuracy_ds_2))
print('[] Log Loss ds 2: ' + str(logLoss_ds_2))

