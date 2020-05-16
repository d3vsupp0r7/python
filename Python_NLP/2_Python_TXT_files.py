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
#-> if we lunch this instruction we have I/O exception:
#   eng_proverb_file.read()

###################
##  WRITE TO FILE
###################
'''
Working with file, we have to manage permissions.
We have the following attribute permissions:
r   : is the default permission
r+  :
w   :
w+  :
a   :
a+  :

'''
#Add
#Using [a] permission type
''' [a] for read
eng_proverb_file = open("files/txt/english-proverbs.txt","a")
print(eng_proverb_file.read());
Traceback (most recent call last):
  File "C:/pythonGithub/python/Python_NLP/2_Python_TXT_files.py", line 104, in <module>
    print(eng_proverb_file.read());
io.UnsupportedOperation: not readable
'''
# [a] fro write
eng_proverb_file = open("files/txt/english-proverbs.txt","a")
new_eng_proverb_01 = "\nPractice makes perfect."
result = eng_proverb_file.write(new_eng_proverb_01)
print("Byte written to file: " + str(result))
eng_proverb_file.close()

print("*) read after writing")
eng_proverb_file = open("files/txt/english-proverbs.txt","a+")
print(eng_proverb_file.readlines())
'''
a+ put the file cursor at the end of file, if we wanto to read using [a+] permission we need to
use seek() method.
'''
print(" *) [a+] read after writing - seek()")
eng_proverb_file.seek(0)
print(eng_proverb_file.readlines())
#
new_eng_proverb_02 = "\nIf you can't beat 'em, join 'em."
result = eng_proverb_file.write(new_eng_proverb_02)
eng_proverb_file.close()

### [CREATE A NEW FILE] - w+ permission
eng_proverb_file = open("files/txt/english-proverbs.txt")
eng_proverb_new_file = open("files/txt/english-new-proverbs.txt","w+")
for i, proverb in enumerate(eng_proverb_file):
    if(i>=3):
        break
    print(proverb)
    eng_proverb_new_file.write(proverb)
# Read new created file
eng_proverb_new_file.seek(0)
print(eng_proverb_new_file.readlines())
eng_proverb_file.close()
eng_proverb_new_file.close()


## ITA Proverb files ##
print("*** [ITALIAN PROVERBS EXAMPLES] ***")
#1. Declare path file to open
ita_proverb_file = open("files/txt/italian-proverbs.txt")
#2.
print(type(ita_proverb_file))
#3.
ita_proverb_file.read()