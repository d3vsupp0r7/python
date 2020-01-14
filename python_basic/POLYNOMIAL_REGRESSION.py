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
