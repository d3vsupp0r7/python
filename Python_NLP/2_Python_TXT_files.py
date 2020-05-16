print("*** Python TXT file management ***")
print("*** [ENGLISH PROVERBS EXAMPLES] ***")
#1. Declare path file to open
eng_proverb_file = open("files/txt/english-proverbs.txt")

'''
It there is into fiel some special caracters, we can use the utf-8 encoding.
eng_proverb_file = open("files/txt/english-proverbs.txt", encoding="utf-8")
'''
#2.
print(type(eng_proverb_file))
#3. Printing readed file
print(eng_proverb_file.read())
print("*) Second READ")
print(eng_proverb_file.read())
'''
Wen we call read the second time we not give any values. This because the reading cursor 
remain at end of file. To manually move to cursor insiede the opened file, we need to use the
seek() method
'''
print("*) File cursor: seek() method")
print("*) THIRD READ - seek()")
eng_proverb_file.seek(0) #0 means to the beginning of file
print(eng_proverb_file.read())
print("*) FOUR READ - seek10: read from )")
eng_proverb_file.seek(10) #Read from 10 position []
print(eng_proverb_file.read())
eng_proverb_file.seek(12) #Read from 12 position []
print(eng_proverb_file.read())
'''
          | 
01234567890123456
Two wrongs don't make a right. [position 0]
           don't make a right. [position 10]
            on't make a right. [position 12]
'''
print("2) readline() and readlines() functions")
print("*) readlines")
eng_proverb_file.seek(0)
for proverb in eng_proverb_file.readlines():
    print("Proverb: %s" % proverb)
#
print("     *) Print removing neline character")
eng_proverb_file.seek(0)
for proverb in eng_proverb_file.readlines():
    print("Proverb: %s" % proverb[:-1])
#
print("     *) Print getting the index")
'''
Functions:
*) enumerate() :
'''
eng_proverb_file.seek(0)
for i,proverb in enumerate(eng_proverb_file.readlines()):
    print("Proverb[%d]: %s" % ( i+1,proverb[:-1]) )

# Without readlines
print("     *) Print Without readlines")
eng_proverb_file.seek(0)
for i,proverb in enumerate(eng_proverb_file):
    print("Proverb[%d]: %s" % ( i+1,proverb[:-1]) )

# IMPO: closng file after used it
eng_proverb_file.close()
'''
IMPO: Error if we close file and try to read it:
After closing a file, we use these code:
eng_proverb_file.seek(0)
print(eng_proverb_file.read())

we have an exception, that inform us that the file is closed

Traceback (most recent call last):
  File "C:/your/python/file_path/2_Python_TXT_files.py", line 65, in <module>
    eng_proverb_file.seek(0)
ValueError: I/O operation on closed file.
'''

## FILE BEST PRACTICE: Open file in context
print("[OPEN FILE INTO A CONTEXT]")
with open("files/txt/english-proverbs.txt") as eng_proverb_file:
    print(eng_proverb_file.read())

# If we read the file outside the context we have the same I/O Exception, we can't access the file.
eng_proverb_file.read()


## ITA Proverb files ##
print("*** [ITALIAN PROVERBS EXAMPLES] ***")
#1. Declare path file to open
ita_proverb_file = open("files/txt/italian-proverbs.txt")
#2.
print(type(ita_proverb_file))
#3.
ita_proverb_file.read()