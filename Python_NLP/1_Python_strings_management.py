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
###
print('#10 - String replace() method')
print("*) Replace example")
str_var = 'This is an example String variable to use for Python Strings processing. Awesome!!! X :)'
print(str_var)
str_var = str_var.replace("Awesome","Excellent")
print(str_var)
###
print('#11 - String find() method')
str_var = 'This is an example String variable to use for Python Strings processing. Awesome!!! X :)'
#
print(str_var.find('T'))
print(str_var[0])
print("*****")
#
print(str_var.find('A'))
print(str_var[73])
print("*****")
#
print(str_var.find('a'))
print(str_var[8])
print("*****")
#
print("*) Substring with find")
# Obtain text from a char until the end
subs_index = str_var.find('v')
print("index: " + str(subs_index) )
str_substring = str_var[subs_index:]
print(str_substring)
# obtain text up to the first point
subs_end_index = str_var.find('.')
print("index: " + str(subs_end_index) )
str_substring = str_var[:subs_end_index]
print(str_substring)
# obtain text from dot to end
subs_end_index = str_var.find('.')
print("index: " + str(subs_end_index) )
str_substring = str_var[subs_end_index:]
print(str_substring)
# no pattern present into string, find() will return -1
print("*) find() no pattern was found (-1) ")
subs_end_index = str_var.find('test')
print("index: " + str(subs_end_index) )
####
print("*) Find from end: rfind() function")
proverb = "Eye for eye, tooth for tooth"
##Get first occurrence of pattern
subs_index = proverb.find("tooth")
print("index: " + str(subs_index) )
print("*) rfind")
subs_index = proverb.rfind("tooth")
print("index: " + str(subs_index) )
##
print("*) Managing spaces into string: strip() function")
proverb_strip_example = "         Eye for eye, tooth for tooth   "
print("Lenght of proverb spaced string: " + str(len(proverb_strip_example) ) )
print(proverb_strip_example.strip())
print("     *) lstrip() ")
proverb_lstrip_example = proverb_strip_example.lstrip()
print(proverb_lstrip_example)
print("Lenght of * lsprip * proverb spaced string: " + str(len(proverb_lstrip_example) ) )
print("     *) rstrip() ")
proverb_rstrip_example = proverb_strip_example.rstrip()
print(proverb_rstrip_example)
print("Lenght of * rsprip * proverb spaced string: " + str(len(proverb_rstrip_example) ) )