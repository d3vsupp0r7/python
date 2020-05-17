'''
NLP - CORE CONCEPTS

Tokens: Tokenization is essentially splitting a phrase, sentence,
paragraph, or an entire text document into smaller units,
such as individual words or terms. Each of these smaller units
are called tokens.

\_> we need to divide by punctuation. every punctuation needs to be separed token.

|_> We need to recognize single entity
As Example:

* We need to create a set of rules. This means that every language have their own rules
that are coden into software pieces called tokenizer.

Python library: NLTK
'''
#
import nltk
'''
This instruction will download software that is necessary to use nltk library.

nltk.download("punkt") -> must be executed the first time
'''
#nltk.download("punkt")
'''
NLTK version: 3.4.5.
'''
print('The nltk version is {}.'.format(nltk.__version__))

ita_textToAnalize_01 = "Cos'è questa fretta? Facciamolo un'altra volta, ti va bene?"
eng_textToAnalize_01 = "What is this haste? Let's do it again, okay?"

ita_tokens = ita_textToAnalize_01.split()
eng_tokens = eng_textToAnalize_01.split()
print(" ** TOKEN SPLIT TYPE **")
print(type(ita_tokens))

print(" ** ITA TOKEN SPLIT **")
print(ita_tokens)
print(" ** ENG TOKEN SPLIT **")
print(eng_tokens)

# STEP 1: USE NLTK library
from nltk import word_tokenize
nltk_ita_tokens = word_tokenize(ita_textToAnalize_01)
nltk_eng_tokens = word_tokenize(eng_textToAnalize_01)
print(" ** NLTK TOKEN TYPE **")
print(type(nltk_ita_tokens))
print("     *) ITA NLTK TOKENS **")
print(nltk_ita_tokens)
print("     *) ENG NLTK TOKEN **")
print(nltk_eng_tokens)

#
print('*) Evaluate ENG complex phrase')
eng_textToAnalize_02 = ("I'd love to visit U.S.A, in particular i want to run with a motorcycle on Route 66 making a classic trip on the road. "
"I'd like to live the emotions like the Easy Rider film, watching the wild nature. But... i'cant. What a nice experience it would be!!!!")
print(eng_textToAnalize_02)
print(" ** NLTK TOKEN TYPE **")
nltk_eng_tokens = word_tokenize(eng_textToAnalize_02)
print("*) Tokens type: ")
print(type(nltk_eng_tokens))
print("*)Tokens of string: " + str(len(nltk_eng_tokens)) )
print("*)Tokens printed: ")
print(nltk_eng_tokens)

#Iterate over tuples
'''for var in nltk_eng_tokens:
    print ("["+str(nltk_eng_tokens.index(var)) + "]",var)
'''

#Iterate over list
print("*)Tokens printed and formatted: ")
for i in range(len(nltk_eng_tokens)):
    print("Token [{}]: {}".format(i + 1, nltk_eng_tokens[i]))

#NLTK: tokenize in phrases
print(" ** NLTK TOKENS IN PHRASES TYPE **")
from nltk.tokenize import sent_tokenize
#5 propositions [?]
print("*) Original Phrase")
print(eng_textToAnalize_02)
'''
Token [1]: I'd love to visit U.S.A, in particular i want to run with a motorcycle on Route 66 making a classic trip on the road.
Token [2]: I'd like to live the emotions like the Easy Rider film, watching the wild nature.
Token [3]: But... i'cant.
Token [4]: What a nice experience it would be!!!
Token [5]: !
'''
nltk_eng_phrases_tokens = sent_tokenize(eng_textToAnalize_02)
print(type(nltk_eng_phrases_tokens))
print("*)Tokens of string: " + str(len(nltk_eng_phrases_tokens)) )
print("*)Tokens printed: ")
print(nltk_eng_phrases_tokens)
print("*)Tokens printed and formatted: ")
#
for i in range(len(nltk_eng_phrases_tokens)):
    print("Token [{}]: {}".format(i + 1, nltk_eng_phrases_tokens[i]))

## STOP WORDS ##
'''
Stop word are common use words that they do not bring any useful information to the text
Stop words are:
*) congiunction:

'''