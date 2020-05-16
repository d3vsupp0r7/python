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

