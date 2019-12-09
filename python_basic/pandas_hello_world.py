import numpy as np
import pandas as pd

'''
Series:
Series is a one-dimensional array, which can hold any type of data, 
such as integers, floats, strings, and Python objects too.
'''
seriesDataExample = pd.Series(np.random.randn(5))
print(seriesDataExample)
'''
The series function creates a pandas series that consists of an index, which is the first column, and the second column consists of random values.
'''
seriesDataExampleWithLabel =pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
print(seriesDataExampleWithLabel)

d = {'A': 10, 'B': 20, 'C': 30}
dExample = pd.Series(d)
print(dExample)

####################################################
# DataFrame
'''
DataFrame is a 2D data structure with columns that can be of different datatypes. 
It can be seen as a table. 
A DataFrame can be formed from the following data structures:
    -) A NumPy array
    -) Lists
    -) Dicts
    -) Series
    -) A 2D NumPy array
'''
## Dataframe creation
dataFrameData1 = {'c1': pd.Series(['A', 'B', 'C']),
     'c2': pd.Series([1, 2., 3., 4.])}
dataFrameObject1 = pd.DataFrame(dataFrameData1)
print(dataFrameObject1)

dataFrameData2 = {'c1': ['A', 'B', 'C', 'D'],
                  'c2': [1, 2.0, 3.0, 4.0]}
dataFrameObject2 = pd.DataFrame(dataFrameData2)
print(dataFrameObject2)

#print("First row: " + dataFrameObject1.head())
## Read first row of dataFrame

# Panel
'''
A Panel is a data structure that handles 3D data.
'''
