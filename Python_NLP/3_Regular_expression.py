#Python - Regular expressions
#Library to process regex
import re

# [PATTERNS]
simple_email_regex = "[a-z0-9]+@[a-z]+\.[a-z]{2,3}"

# Code
'''
|        |         |         |         | 
012345678901234567890123456789012345678901
This is an example to use regex in Python.
'''
textToAnalize = "This is an example to use regex in Python."
string_to_find = "example"

search_result = re.search(string_to_find,textToAnalize)
print(type(search_result))
## Analize result
print("*) Text to analyze: \n\t" + textToAnalize)
print("*) Length text to analyze: " + str(len(textToAnalize)) )
print("*) Pattern to find: \n\t" + string_to_find)
print("*) Length pattern to find: " + str(len(string_to_find)) )
#Search result
print("*) span() method result")
print(search_result.span())
print("*) re library: get only start index")
print(search_result.start())
print("*) re library: get only end index")
print(search_result.end())
print("*) re library: extract/slicing ")
textToAnalize_slicing_1 = textToAnalize[search_result.start():search_result.end()]
print(textToAnalize_slicing_1)
# Use of group() function for slicing
print("*) Slicing using group() method")
print(search_result.group())
###
print("REGEX Example - Phone number")
textToAnalyze_01 = ("For information call this example number 123 456 7890.")

rgx_pattern_simple_number_phone = r"\d\d\d \d\d\d \d\d\d\d"
rgx_simple_result_01 = re.search(rgx_pattern_simple_number_phone, textToAnalyze_01)
print("Regex result for simple phone number: " + rgx_simple_result_01.group())
# Example 2: using mixed chars
print("*) Example 2: using mixed chars")
textToAnalyze_01 = ("For information call this example number 123-456-7890.")
rgx_pattern_simple_number_phone = r"\d\d\d-\d\d\d-\d\d\d\d"
rgx_simple_result_01 = re.search(rgx_pattern_simple_number_phone, textToAnalyze_01)
print("Regex result for simple phone number: " + rgx_simple_result_01.group())
#Example 3 - regex optimized
print("*) Example 3 - regex optimized")
textToAnalyze_01 = ("For information call this example number 123-456-7890.")
rgx_pattern_simple_number_phone = r"\d{3}-\d{3}-\d{4}"
rgx_simple_result_01 = re.search(rgx_pattern_simple_number_phone, textToAnalyze_01)
print("Regex OPTIMIZATION: result for simple phone number: " + rgx_simple_result_01.group())
#Example 4 - grouping te patterns
print("*) Example 4 - grouping te patterns")
'''
For grouping we need to add parentesis to regex definition
'''
textToAnalyze_01 = ("For information call this example number +39 123-456-7890.")
rgx_pattern_simple_number_phone = r"(\d{2}) (\d{3})-(\d{3}-\d{4})"
rgx_simple_result_01 = re.search(rgx_pattern_simple_number_phone, textToAnalyze_01)
print("Regex grouping: result for simple phone number: " + rgx_simple_result_01.group())
print("Out 1 - group(): " + rgx_simple_result_01.group())
print("Out 1 - group(0): " + rgx_simple_result_01.group(0))
print("Out 1 - group(1): " + rgx_simple_result_01.group(1))
print("Out 1 - group(2): " + rgx_simple_result_01.group(2))
print("Out 1 - group(3): " + rgx_simple_result_01.group(3))
#
'''
If we go with index outside group index, a exception will be thrown
Traceback (most recent call last):
  File "C:/pythonGithub/python/Python_NLP/3_Regular_expression.py", line 73, in <module>
    print("Out 1 - group(4): " + rgx_simple_result_01.group(4))
IndexError: no such group

-> print("Out 1 - group(4): " + rgx_simple_result_01.group(4))
this expression generate exception.
'''
#Obtaining multiple result
textToAnalyze_02 = ("For information call this example number +39 123-456-7890. For coding information you also call "
                    "the following number: +39 098-765-4321")
rgx_pattern_simple_number_phone = r"(\d{2}) (\d{3})-(\d{3}-\d{4})"
multiple_search_result = re.findall(rgx_pattern_simple_number_phone, textToAnalyze_02)
print(type(multiple_search_result))
print(len(multiple_search_result) )
#Progessing/accessing to list elements into python
print(multiple_search_result)
print(multiple_search_result[0])
print(multiple_search_result[0][0])
print(multiple_search_result[0][1])
print(multiple_search_result[0][2])
#
print(multiple_search_result[1])
print(multiple_search_result[1][0])
print(multiple_search_result[1][1])
print(multiple_search_result[1][2])
#
'''
Accessing to a list with wrong index:

Traceback (most recent call last):
  File "C:/pythonGithub/python/Python_NLP/3_Regular_expression.py", line 100, in <module>
    print(multiple_search_result[2])
IndexError: list index out of range

-> print(multiple_search_result[2]) this will trhrow exception
'''


