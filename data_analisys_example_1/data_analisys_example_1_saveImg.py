from itertools import groupby

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Import mode function:
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
'''
from sklearn import cross_validation, metrics
cross_validation is deprecated
'''
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import metrics

#%matplotlib inline
from statsmodels.compat import numpy

'exec(%matplotlib inline)'
import warnings # Ignores any warning

from matplotlib.pyplot import figure

warnings.filterwarnings("ignore")

train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")

# Graph data constants
graph_index = 1
img_ext = '.png'
heatmap_img_type_ref = 'heatmap_'
histogram_img_type_ref = 'histogram_'

##

print("** Columns info **")
print(train.columns)
print("** Head info **")
print(train.head())
print("** Info **")
print(train.info())
print("** Describe data **")
print(train.describe())

# Check for duplicates
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
# Important note: to save picture file, first use savefig() method and then use show() to show graph.
#print("Graph index: " + graph_index)
print( "{} - {}".format("Graph index: ", graph_index) )
plt.savefig('data/fig_1.png')
plt.show()
# Python not use this operator (++): graph_index++
graph_index += 1
# print("Graph index increment: " + graph_index)
print("{} - {}".format("Graph index increment: ", graph_index) )
plt.savefig('data/fig_' + str(graph_index) + '.png')

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
# Working but overlay on same sns graph -> sns.countplot(train.Item_Fat_Content)

#Analize the relationship between "taregt variable" and predictors
plt.figure(figsize=(12,7))
plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weight and Item_Outlet_Sales Analysis")
plt.plot(train.Item_Weight, train["Item_Outlet_Sales"],'.', alpha = 0.3)
plt.show()

#Analisys on item visibility on store
plt.figure(figsize=(12,7))
plt.xlabel("Item_Visibility")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Visibility and Item_Outlet_Sales Analysis")
plt.plot(train.Item_Visibility, train["Item_Outlet_Sales"],'.', alpha = 0.3)
plt.show()

# Impact: Outlet_Establishment_Year and Item_Outlet_Sales analysis
print("** Impact: Outlet_Establishment_Year and Item_Outlet_Sales analysis ")
Outlet_Establishment_Year_pivot = train.pivot_table(index='Outlet_Establishment_Year', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Establishment_Year_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Establishment_Year")
plt.ylabel("Sqrt Item_Outlet_Sales")
plt.title("Impact of Outlet_Establishment_Year on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()

# Impact: Item_Fat_Content onItem_Outlet_Sales analysis
print("** Impact: Item_Fat_Content onItem_Outlet_Sales analysis ")
Item_Fat_Content_pivot = train.pivot_table(index='Item_Fat_Content', values="Item_Outlet_Sales", aggfunc=np.median)
Item_Fat_Content_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Item_Fat_Content")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Item_Fat_Content on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()

# Impact: Outlet_Identifier on Item_Outlet_Sales analysis
print("** Outlet_Identifier on Item_Outlet_Sales analysis ")
Outlet_Identifier_pivot = train.pivot_table(index='Outlet_Identifier', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Identifier_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Identifier")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Identifier on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()

# ? no effect
# train.pivot_table(values='Outlet_Type', columns='Outlet_Identifier',aggfunc=lambda x:x.mode())
# Impact: Outlet_Size on Item_Outlet_Sales analysis
print("** Impact: Outlet_Size on Item_Outlet_Sales analysis ")
Outlet_Size_pivot = train.pivot_table(index='Outlet_Size', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Size_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Size")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Size on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()

# Impact: Outlet_Type on Item_Outlet_Sales analysis
print("** Impact: Outlet_Type on Item_Outlet_Sales analysis ")
Outlet_Type_pivot = train.pivot_table(index='Outlet_Type', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Type_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Type ")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Type on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()

# Impact: Outlet_Location_Type  on Item_Outlet_Sales analysis
print("** Impact: Outlet_Location_Type  on Item_Outlet_Sales analysis ")
Outlet_Location_Type_pivot = train.pivot_table(index='Outlet_Location_Type', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Location_Type_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Location_Type")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Location_Type on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()

# Concat Dataset
print("** Dataset Concatenation ")
train['source']='train'
test['source']='test'
data = pd.concat([train,test], ignore_index = True)
print(train.shape, test.shape, data.shape)

# Check the percentage of null values per variable
print("** Check the percentage of null values per variable ")
data.isnull().sum()/data.shape[0]*100 #show values in percentage
print(data.isnull().sum()/data.shape[0]*100)

# mean of NaN/Missing values
#aggfunc is mean by default! Ignores NaN by default
# Imputing Missing Values
'''
pivot_table() is used to calculate, aggregate, and summarize your data.
'''
print("** mean of NaN/Missing values - for values Item_Weight")
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')
print(item_avg_weight)
print(data[:][data['Item_Identifier'] == 'DRI11'])

''' ------ '''

# Example of function declaration
## Scope to assign mean to NaN values ?
def impute_weight(cols):
    Weight = cols[0]
    Identifier = cols[1]

    if pd.isnull(Weight):
        return item_avg_weight['Item_Weight'][item_avg_weight.index == Identifier]
    else:
        return Weight

print('Orignal #missing: %d' % sum(data['Item_Weight'].isnull()))
data['Item_Weight'] = data[['Item_Weight', 'Item_Identifier']].apply(impute_weight, axis=1).astype(float)
print('Final #missing: %d' % sum(data['Item_Weight'].isnull()))
''' ------ '''
# Mean for text values
## Determing the mode for each
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=lambda x:x.mode())
print(outlet_size_mode)

def impute_size_mode(cols):
    Size = cols[0]
    Type = cols[1]
    if pd.isnull(Size):
        return outlet_size_mode.loc['Outlet_Size'][outlet_size_mode.columns == Type][0]
    else:
        return Size

print ('Orignal #missing: %d'%sum(data['Outlet_Size'].isnull()))
data['Outlet_Size'] = data[['Outlet_Size','Outlet_Type']].apply(impute_size_mode,axis=1)
print ('Final #missing: %d'%sum(data['Outlet_Size'].isnull()))

##############################################
############  FEATURE ENGINE SECTION #########
print("###  FEATURE ENGINE SECTION ###")
#Creates pivot table with Outlet_Type and the mean of #Item_Outlet_Sales. Agg function is by default mean()
print("*** FEATURE ENGINEERING SECTION ***")
print("* Data relation between Outlet_Type and Item_Outlet_Sales")
print( data.pivot_table(values='Item_Outlet_Sales', columns='Outlet_Type') )

print("* Item_Visibility analysis")
visibility_item_avg = data.pivot_table(values='Item_Visibility',index='Item_Identifier')
def impute_visibility_mean(cols):
    visibility = cols[0]
    item = cols[1]
    if visibility == 0:
        result = visibility_item_avg['Item_Visibility'] [visibility_item_avg.index == item]
        return result
    else:
        return visibility

print ('Original #zeros: %d'%sum(data['Item_Visibility'] == 0))
data['Item_Visibility'] = data[['Item_Visibility','Item_Identifier']].apply(impute_visibility_mean,axis=1).astype(float)
print ('Final #zeros: %d'%sum(data['Item_Visibility'] == 0))

print('* Determine the years of operation of a store')
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()
print(data['Outlet_Years'].describe())

print('## Combination of feature example based on Item_Type ##')

#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()
print(data['Item_Type_Combined'].value_counts())

print('## Renaming of feature based on Item_Fat_Content ##')

#Change categories of low fat:
print('Original Categories:')
print(data['Item_Fat_Content'].value_counts())
print('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                            'low fat':'Low Fat'})
print(data['Item_Fat_Content'].value_counts())

print('## Subcategory of feature based on Item_Fat_Content ##')
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()
print(data['Item_Fat_Content'].value_counts())

##############################################
############  FEATURE TRANSFORMATION #########

'''
We can create a new variable that show us the importance given to a product in a given store according to the mean of
significance given to the same product in all other stores.
'''
func = lambda x: x['Item_Visibility']/visibility_item_avg['Item_Visibility'][visibility_item_avg.index == x['Item_Identifier']][0]
data['Item_Visibility_MeanRatio'] = data.apply(func,axis=1).astype(float)
data['Item_Visibility_MeanRatio'].describe()
print(data['Item_Visibility_MeanRatio'].describe())

## Manage categorical data - Example
le = LabelEncoder()#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']

for i in var_mod:
    data[i] = le.fit_transform(data[i])
    print(data[i])

##############################################
############    EXPORTING DATA       #########
#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)
#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]
#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)
#Export files as modified versions:
train.to_csv("data/train_modified.csv",index=False)
test.to_csv("data/test_modified.csv",index=False)

##############################################
############    MODEL BUILDING       #########
train_df = pd.read_csv('data/train_modified.csv')
test_df = pd.read_csv('data/test_modified.csv')

# Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier', 'Outlet_Identifier']

print("######### Init model fit analysis #########")
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    # Remember the target had been normalized
    Sq_train = (dtrain[target]) ** 2

    # Perform cross-validation:
#    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], Sq_train, cv=20, scoring='neg_mean_squared_error')
    ## ? to verify sobstutution of oldes lib to the new
    cv_score = cross_val_score(alg, dtrain[predictors], Sq_train, cv=20,scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))

    # Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(Sq_train.values, dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (
    np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])

    # Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)

print("### 1. Model fit analysis - Linear Regression Model ###")
from sklearn.linear_model import LinearRegression
LR = LinearRegression(normalize=True)
predictors = train_df.columns.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'])
modelfit(LR, train_df, test_df, predictors, target, IDcol, 'LR.csv')
