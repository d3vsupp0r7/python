# importing pandas and numpy
import pandas as pd
import numpy as np

# crete a sample dataframe
data = pd.DataFrame({
    'age' :     [ 10, 22, 13, 21, 12, 11, 17],
    'section' : [ 'A', 'B', 'C', 'B', 'B', 'A', 'A'],
    'city' :    [ 'Gurgaon', 'Delhi', 'Mumbai', 'Delhi', 'Mumbai', 'Delhi', 'Mumbai'],
    'gender' :  [ 'M', 'F', 'F', 'M', 'M', 'M', 'F'],
    'favourite_color' : [ 'red', np.NAN, 'yellow', np.NAN, 'black', 'green', 'red']
})

# view the data
print(data);
# loc example
# select all rows with a condition
print("[loc: select all rows with a condition]")
print(data.loc[data.age >= 15]);
# select with multiple conditions
print("[loc: select with multiple conditions]")
print(data.loc[(data.age >= 12) & (data.gender == 'M')]);
#Select a range of rows using loc
print("[loc: select a range of rows using loc]")
#slice
print(data.loc[1:3]);
# select few columns with a condition
print("[loc: select few columns with a condition]")
print(data.loc[(data.age >= 12), ['city', 'gender']]);

# select rows with indexes
print('*** ILOC ***')
print("[iloc: select rows with indexes]")
print(data.iloc[[0,2]]);