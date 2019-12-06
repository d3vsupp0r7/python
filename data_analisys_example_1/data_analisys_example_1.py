import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from statsmodels.compat import numpy

'exec(%matplotlib inline)'
import warnings # Ignores any warning

from matplotlib.pyplot import figure

warnings.filterwarnings("ignore")

train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")

print("** Columns info **")
print(train.columns)
print("** Head info **")
print(train.head())
print("** Info **")
print(train.info())
print("** Describe data **")
print(train.describe())

#Check for duplicates
print("** Check for duplicates **")
idsUnique = len(set(train.Item_Identifier))
idsTotal = train.shape[0]
idsDupli = idsTotal - idsUnique
print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")

print("** Exploratory Data Analysis (EDA)")
print("*** Univariate Distribution")
print("*** Distribution of the target variable : Item_Outlet_Sales")
#plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,7))
sns.distplot(train.Item_Outlet_Sales, bins = 25)
plt.ticklabel_format(style='plain', axis='x', scilimits=(0,1))
plt.xlabel("Item_Outlet_Sales")
plt.ylabel("Number of Sales")
plt.title("Item_Outlet_Sales Distribution")
plt.show()

# Skew and kurtosis
print("** Skew and kurtosis")
print ("Skew is:", train.Item_Outlet_Sales.skew())
print("Kurtosis: %f" % train.Item_Outlet_Sales.kurt())

# Print numeric datatypes
print("** Numeric datataypes")
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes
print(numeric_features.dtypes)
#plt.matshow(numeric_features.dtypes)
#plt.show()
#
corr = numeric_features.corr()
print(numeric_features.corr())
plt.matshow(numeric_features.corr())
plt.show()

# plot correlation matrix
fig = plt.figure()
fig.suptitle('Correlation Matrix', fontsize=20)
ax = fig.add_subplot(111)
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
# Retrive column names
ax.set_xticklabels(list(train.columns))
ax.set_yticklabels(list(train.columns))
plt.show()

print("** Sort correlation output and order it descending, [MAX->to->MIN]")
print(corr['Item_Outlet_Sales'].sort_values(ascending=False))

# ? no effect
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True);
#heatmap = sns.heatmap(corr,annot=True);

## Manage Categorical Predictors ?
#sns.countplot(train.Item_Fat_Content)
#plt.show(train.Item_Fat_Content)

#Analize the relationship between "taregt variable" and predictors
plt.figure(figsize=(12,7))
plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weight and Item_Outlet_Sales Analysis")
plt.plot(train.Item_Weight, train["Item_Outlet_Sales"],'.', alpha = 0.3)
plt.show()