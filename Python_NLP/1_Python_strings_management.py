str_var = 'This is an example String variable to use for Python Strings processing. Awesome!!!'
#
print('#01 - ')
print(str_var)
print( type(str_var) )
'''
type : show type of data:
<class 'str'>
'''
print('#02 - Access to char position of string')
print(str_var[0])
print(str_var[1])
print(str_var[2])
print(str_var[3])
#Last char
print(str_var[-1])
print('#03 - Slicing')
'''
slicing: obtain part of string
*) From start to end-1
str_var[0:4]
*) Last n character
print(str_var[:-3])
'''

print(str_var[0:4])
#Last 3 char
print(str_var[:-3])

print('#04 - String immutable in python')
#str_var[0] = 'C'
'''
 str_var[0] = 'C'
TypeError: 'str' object does not support item assignment

Decomment line 31 to see errors, otherwise pytnon script not run until the end.
'''
print('#05 - Modify a string variable')
#Concat
str_var = "MOD_"+str_var;
print(str_var)
print('#06 - String manipulation')
str_var = 'This is an example String variable to use for Python Strings processing. Awesome!!!'
# All string char lower
print(str_var.lower())
# All string char upper
print(str_var.upper())
#capitalize: Only the first char of string uppercase.
str_var = 'this is an TEST example String variable to use for Python Strings processing. Awesome!!!'
print(str_var.capitalize())
###
print('#07 - String splitting')
str_var = 'This is an example String variable to use for Python Strings processing. Awesome!!!'
#Split method
print('** split() method')
word_list = str_var.split()
print("*) Type returned by split method: ")
print( type(word_list) )
print("*) Output of split method: ")
print(word_list)
#Split with argument
print("*) Split words with a specific char to indicate the split.")
str_to_split = "Python split string example. We can split multiple strings. Let's do it. Awesome"
sentences = str_to_split.split(".")
print(sentences)
##
print('#08 - String join() method')
# join method: join a splitted array of words into a string
'''
The join method: 
define a string, using it to call the join method and pass a list to join.
The string on wich we call the join method represent the separated values used for join the string.
'''
print("** Join method")
str_merged_from_list = " ".join(word_list)
print(str_merged_from_list)
str_merged_from_list = "_SPACE_".join(word_list)
print(str_merged_from_list)
print('#09 - String count() method')
print("*) Count occurrence of wod into a string")
str_var = 'This is an example String variable to use for Python Strings processing. Awesome!!! X :)'
print(str_var.count("X"))
print(str_var.count("a"))
print(str_var.count("A"))
'''
count() method is case sensitive.
'''
#Find a word
print(str_var.count("String"))
# 0 if string not contais the char
print(str_var.count("Z"))
print('#09 - String replace() method')