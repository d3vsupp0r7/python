# 1. Use a defined class inside a external file
from EmployeeClassExample import Employee

'''
if we use only

import EmployeeClassExample

we get:

NameError: name 'Employee' is not defined
'''

#id, name,surname,gender, city, salary
emp1 = Employee(1,'aName','aSurname','M','aCity',2000.00)

print(emp1.fullname())