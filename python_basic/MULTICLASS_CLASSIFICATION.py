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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_digits
'''
This example uses the sklearn digit dataset to exlore the multiclass classification problem.
This dataset represent the all digits,0 to 9, digitally scanned.
The objective is to try the right classification for the images, that are rapresented as pixel values on 
gray scale from 0 to 255.
'''
enable_digit_printing = "n"

pd_dataset = load_digits()

X = pd_dataset.data
Y = pd_dataset.target

print('*) DS Info')
print('-) DS shape')
print(X.shape)
print('-) DS classes')
print(np.unique(Y))

# print the example images
if enable_digit_printing.lower() in ['y', 'yes']:
    for i in range(0,10):
        '''
        '''
        fig = plt.figure(figsize=(12,10) )
        pic_matrix = X[Y==i][0].reshape([8,8])
        plt.imshow(pic_matrix, cmap="gray")
        fig.suptitle('Image of number: ' + str(i), fontsize=20)
        plt.show()

# Create set for train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=0)
'''
Normalize columns
'''
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

'''
Apply the model for multiclass classification.
By default the sklearn library can manage the multiclass classification without any additional parameters.

'''

lr = LogisticRegression()
lr.fit(X_train,Y_train)

Y_pred = lr.predict(X_test)
Y_pred_proba = lr.predict_proba(X_test)

print('*** Calculate the performance of models')
accuracy = accuracy_score(Y_test, Y_pred)
print('Accuracy: ' + str(accuracy))

logLoss = log_loss(Y_test, Y_pred_proba)
print('Log Loss: ' + str(logLoss))

# Matrix of confusion
print('*** Print Confusion Matrix')
from sklearn.metrics import confusion_matrix
'''
How to read confusion matrix?

cols: predicted class
rows: correct class

'''
cm = confusion_matrix(Y_test,Y_pred)
from tabulate import tabulate
print(tabulate(cm))

## Using seaborn to plot confusion matrix
import seaborn as sns

plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True)
plt.ylabel("Correct class")
plt.xlabel("Predicted class")
plt.show()

'''
Optimize the heatmap graph drawing
'''
plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True, cmap="Blues_r", linewidths=.5, square=True)
plt.ylabel("Correct class")
plt.xlabel("Predicted class")
plt.show()
