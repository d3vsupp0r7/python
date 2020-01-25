#
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

## sklearn.datasets
from sklearn.datasets import load_digits

'''
IMPO: This example will use the default dataset of sklearn library for digit.

'''
pd_dataset = load_digits()

X = pd_dataset.data
Y = pd_dataset.target

print('*) DS Info')
print('-) DS shape')
print(X.shape)
print('-) DS classes')
print(np.unique(Y))

# Create set for train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=0)

#
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

#
from sklearn.neighbors import KNeighborsClassifier
print('** K Neighbors Classifier **')
knn = KNeighborsClassifier()
out = knn.fit(X_train,Y_train)

#
Y_pred_train = knn.predict(X_train)
Y_prob_train = knn.predict_proba(X_train)

Y_pred_test = knn.predict(X_test)
Y_prob_test = knn.predict_proba(X_test)

print('** K Neighbors Classifier: Metrics **')
print('-) Calculate the performance of models TRAIN DS')
accuracy_train = accuracy_score(Y_train, Y_pred_train)
print('Accuracy (train): ' + str(accuracy_train))
print('Accuracy (train): %.4f' % (accuracy_train))
logLoss_train = log_loss(Y_train, Y_prob_train)
print('Log Loss (train): ' + str(logLoss_train))
print('Log Loss (train): %.4f' % (logLoss_train))

print('-) Calculate the performance of models TEST DS')
accuracy_test = accuracy_score(Y_test, Y_pred_test)
print('Accuracy (test): ' + str(accuracy_test))
logLoss_test = log_loss(Y_test, Y_prob_test)
print('Log Loss (test): ' + str(logLoss_test))

## Hyperparameter K analysis ##
print("*** Analisysis with Ks values ***")
Ks = [1,2,3,4,5,7,10,12,15,20]
for K in Ks:
    print("*******************************")
    print("**           START           **")
    print("*** K value: " + str(K) + " ***")
    knn = KNeighborsClassifier(n_neighbors=K)
    out = knn.fit(X_train, Y_train)
    #
    Y_pred_train = knn.predict(X_train)
    Y_prob_train = knn.predict_proba(X_train)

    Y_pred_test = knn.predict(X_test)
    Y_prob_test = knn.predict_proba(X_test)

    print('** K Neighbors Classifier: Metrics **')
    print('-) Calculate the performance of models TRAIN DS')
    accuracy_train = accuracy_score(Y_train, Y_pred_train)
    print('Accuracy (train): ' + str(accuracy_train))
    logLoss_train = log_loss(Y_train, Y_prob_train)
    print('Log Loss (train): ' + str(logLoss_train))

    print('-) Calculate the performance of models TEST DS')
    accuracy_test = accuracy_score(Y_test, Y_pred_test)
    print('Accuracy (test): ' + str(accuracy_test))
    logLoss_test = log_loss(Y_test, Y_prob_test)
    print('Log Loss (test): ' + str(logLoss_test))
    print("**           END           **")
    print("*******************************")

print("*******************************")
print("** IN DEPTH ANALYSIS FOR A K PPARAMETER WITH PLOT OF RESULTS **")
fixed_k_example = 3
print("** K parameters value: " + str(fixed_k_example) + " **")
knn = KNeighborsClassifier(n_neighbors=fixed_k_example)
out = knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)
for i in range(0,len(X_test)):
    if(Y_pred[i] != Y_test[i]):
        print("Number %d classified with %d" % (Y_test[i],Y_pred[i]))
        plt.imshow(X_test[i].reshape([8,8]), cmap="gray" )
        plt.show()
