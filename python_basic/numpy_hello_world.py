import numpy as np

n_array = np.array([[0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11]])
#
print(n_array.ndim)
# Print the Number of RowxColumn matrix of array
print(n_array.shape)
# Print the number of elements into the array
print(n_array.size)
# Print the type of element into array
print(n_array.dtype.name)

# Array operations
# Subtraction
a = np.array( [11, 12, 13, 14])
b = np.array( [ 1, 2, 3, 4])
c = a - b
print(c)

# Matrix multiplication
A1 = np.array([[1, 1],[0, 1]])
A2 = np.array([[2, 0],[3, 4]])
A3 = A1 * A2
print(A3)
## dot product:
A3_dot_product = np.dot(A1,A2)
print(A3_dot_product)

# Operation on array
## Select an element of array
element_selected = n_array[0,1]
print(element_selected)
## Range selection
range_selected = n_array[ 0 , 0:3 ]
print(range_selected)

## Row Selection
row_selection = n_array[ 0 , : ]
print(row_selection)

## Column Selection
column_selection = n_array[ : , 1 ]
print(column_selection)