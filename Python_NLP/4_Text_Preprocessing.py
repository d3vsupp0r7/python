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
"I'd like to live the emotions like the Easy Rider film, watching the wild nature. But... i'can't. What a nice experience it would be!!!!")
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
*) adverbs
*) congiunction
*) prepositions
*) pronouns
*) common verbs

So, if stop word, don't make valid a qualitative information on text, it's best practice to remove them 
when we analyze text.

'''

from nltk.corpus import stopwords
'''
NTLK have library for managing stopwords. In order to use this library, we need to download the appropriate
package, with instruction:

nltk.download('stopwords')

** THIS is one-time operation
'''
#nltk.download('stopwords')

#
print("** STOPWORDS RECAP **")
print("** ENG **")
nltk_eng_stopwords = stopwords.words('english')
#
print("*) English NLTK stopwords managed: " + str(len(nltk_eng_stopwords) ) )
print("*) First 10 English NLTK stopwords: ")
print(nltk_eng_stopwords[:10])
print("*) First 100 English NLTK stopwords: ")
print(nltk_eng_stopwords[:100])
##
print("** ITA **")
nltk_ita_stopwords = stopwords.words('italian')
print("*) Italian NLTK stopwords managed: " + str(len(nltk_ita_stopwords) ) )
print("*) First 10 Italian NLTK stopwords: ")
print(nltk_ita_stopwords[:10])
print("*) First 100 Italian NLTK stopwords: ")
print(nltk_ita_stopwords[:100])
##
#ENG - Remove stopwords

print("** ENG - REMOVING STOPWORDS **")
print(eng_textToAnalize_02)
nltk_eng_stopwords = stopwords.words("english")
nltk_eng_phrases_tokens = word_tokenize(eng_textToAnalize_02)
print(nltk_eng_phrases_tokens)

eng_tokes_filtered = []
eng_tokes_removed_filtered = []

for eng_token in nltk_eng_phrases_tokens:
    if (eng_token.lower() not in nltk_eng_stopwords):
        eng_tokes_filtered.append(eng_token)
    else:
        eng_tokes_removed_filtered.append(eng_token)

print('*) ENG TOKEN - VALID FILTERED')
print(eng_tokes_filtered)
print('*) ENG TOKEN - INVALID FILTERED')
print(eng_tokes_removed_filtered)


print("** ITA - REMOVING STOPWORDS **")
ita_textToAnalize_01 = ("Mi piacerebbe visitare gli Stati Uniti, in particolare voglio correre con una moto sulla Route 66 per fare un classico viaggio su strada. "
"Mi piacerebbe vivere le emozioni come il film di Easy Rider, guardando la natura selvaggia. Ma ... non posso. Che bella esperienza sarebbe !!!!")

nltk_ita_stopwords = stopwords.words("italian")
nltk_ita_phrases_tokens = word_tokenize(ita_textToAnalize_01)
print(ita_textToAnalize_01)

ita_tokens_filtered = []
ita_tokens_removed_filtered = []

for ita_token in nltk_ita_phrases_tokens:
    if (ita_token.lower() not in nltk_ita_stopwords):
        ita_tokens_filtered.append(ita_token)
    else:
        ita_tokens_removed_filtered.append(ita_token)

print('*) ITA TOKEN - VALID FILTERED')
print(ita_tokens_filtered)
print('*) ITA TOKEN - INVALID FILTERED')
print(ita_tokens_removed_filtered)

### STEMMING ###
print("**********************")
print("** STEMMING ")
print("**********************")
'''
STEMMING (radice): 
Stemming, we truncate final part of words based on a set of rules.
As example, 
Stem (root) is the part of the word to which you add inflectional (changing/deriving) affixes such as 
(-ed,-ize, -s,-de,mis). So stemming a word or sentence may result in words that are not actual words. 
Stems are created by removing the suffixes or prefixes used with a word.
IMPO: Removing suffixes from a word is called Suffix Stripping

(*-ITA) STEMMING: analisi di  radice e desinenza di una parola.

The stemming technique is useful because it can reduce to more unique words our test to analyze.
This reduce complexity end computational costs.  

NLTK Library offer stemming tools.
'''
print("**********************")
print("** STEMMING - ENG LANGUAGE")
print("**********************")
from nltk.stem.porter  import PorterStemmer
print("*** NLTK - Stemmer algorithms")
print("*** ENG - Porter Stemmer algorithm ***")
'''

'''
print("Original phrase")
eng_textToAnalize_03 = "I'd loved to visit U.S.A, in particular running with a motorcycle on Route 66 making a classic" \
                       " trip on the road. " \
                       "I loved to be going around with my friends, watching the wild nature. " \
                       "I've watched a lot of things, visiting and watching a lot of places, cooking on the roads. " \
                       "We also have runned a lot of dangers expecially the night on desert places with only the breeze " \
                       "to keep us company. " \
                       "An ice cold beer, a good old blues song and the dreams become true."
print(eng_textToAnalize_03)
#
eng_tokens = word_tokenize(eng_textToAnalize_03)
eng_porter_stemmer = PorterStemmer()
word_to_stem = "watching"
stem_out = eng_porter_stemmer.stem(word_to_stem)
print("*) Stemmming: output for word: " + word_to_stem)
print(stem_out)

#Process stem analisys
eng_tokens_stem = []
print("TOKEN\t\tSTEM")
for eng_token in eng_tokens:
    eng_token_stem_radix = eng_porter_stemmer.stem(eng_token)
    eng_tokens_stem.append(eng_token_stem_radix)
    print("%s\t\t%s" % (eng_token, eng_token_stem_radix) )

print("*** ENG - Snowball Stemmer algorithm ***")
from nltk.stem.snowball import SnowballStemmer
print("Original phrase")
print(eng_textToAnalize_03)
eng_tokens = word_tokenize(eng_textToAnalize_03)
eng_snowball_stemmer = SnowballStemmer("english")
#Process snowball stem analisys
eng_tokens_snowball_stem = []
print("TOKEN\t\tSTEM")
for eng_token in eng_tokens:
    eng_token_stem_radix = eng_snowball_stemmer.stem(eng_token)
    eng_tokens_snowball_stem.append(eng_token_stem_radix)
    print("%s\t\t%s" % (eng_token, eng_token_stem_radix) )



print("*** ENG - Lancaster Stemmer algorithm ***")
'''

'''
from nltk.stem import LancasterStemmer
print("Original phrase")
print(eng_textToAnalize_03)
eng_tokens = word_tokenize(eng_textToAnalize_03)
eng_lancaster_stemmer = LancasterStemmer()
#Process snowball stem analisys
eng_tokens_lancaster_stem = []
print("TOKEN\t\tSTEM")
for eng_token in eng_tokens:
    eng_token_stem_radix = eng_lancaster_stemmer.stem(eng_token)
    eng_tokens_lancaster_stem.append(eng_token_stem_radix)
    print("%s\t\t%s" % (eng_token, eng_token_stem_radix) )

print("**********************")
print("** LEMMATIZATION ")
print("**********************")
'''
LEMMATIZATION

LEMMATIZATION is a process to reduce word from they inflected form to a canonical. form.
The canonical form is called "lemma". 
This type of approach is more efficient.
Lemmatizatition make analysis on a linguistic methodology. Analizyng the mean of 
phrases with terms that means different things is important.
This approach will be used as best practice when analyzing texts.

As example:
'Caring' -> Lemmatization -> 'Care'
'Caring' -> Stemming -> 'Car'

'''

print("**********************")
print("** LEMMATIZATION - ENG LANGUAGE ")
print("**********************")
'''
Lemmatization using NLTK library

To use a lemmer class of NLTK, we need to download additional NLTK package called
WordNetLemmetizer.

We can do this using the following instruction:

nltk.download('wordnet')

'''
from nltk.stem import WordNetLemmatizer

print("Original phrase")
print(eng_textToAnalize_03)
#1. Get tokens
eng_tokens = word_tokenize(eng_textToAnalize_03)
eng_lemmatizer = WordNetLemmatizer()

word_to_lemmatize = "watching"
lemma_out = eng_lemmatizer.lemmatize(word_to_lemmatize)
print("*) Lemmatization: output for word: " + word_to_stem)
print(lemma_out)

print("*) First 10 English NLTK tokens: ")
print(eng_tokens[:10])
print("*** ENG NLTK LEMMA PROCESSING ***")
print("*) ENG NLTK LEMMA PROCESSING: FIRST ITERATION")
'''
If we use WordNetLemmatizer as default, the text we are processing are treated as if it contains all nouns,
as parts of the text are not recognized by default.
'''
eng_tokens_lemmas = []
print("TOKEN\t\tLEMMA")
for eng_token in eng_tokens:
    eng_token_lemma = eng_lemmatizer.lemmatize(eng_token)
    eng_tokens_lemmas.append(eng_token_lemma)
    print("%s\t\t%s" % (eng_token, eng_token_lemma) )
print("*) ENG NLTK LEMMA PROCESSING: SECOND ITERATION")
'''
Manually define part of text => Part Of Speech => POS
n = nouns
v = verbs
a = adjective
r = adverbs

'''
word_to_lemmatize = "watching"
eng_single_token_tuple = [("watching","v")]
eng_single_token_tuple_element = eng_single_token_tuple[0]
lemma_out = eng_lemmatizer.lemmatize(eng_single_token_tuple_element[0],pos=eng_single_token_tuple_element[1])
print("*) Lemmatization [with POS] out for word: " + word_to_stem)
print(lemma_out)

eng_textToAnalize_03_paragraph_1 =  ("I'd loved to visit U.S.A, in particular running with a motorcycle on Route 66 making a classic" \
" trip on the road. ")

#
nltk_eng_phrases_tokens = word_tokenize(eng_textToAnalize_03)
print(nltk_eng_phrases_tokens)

print("*) Lemmatization [POS FULL TEST]: 1. Tokens printed and formatted: ")
for i in range(len(nltk_eng_phrases_tokens)):
    #print("Token [{}]: {}".format(i + 1, nltk_eng_tokens[i]))
    print("(\""+nltk_eng_phrases_tokens[i]+"\",\"\")," )

## eng_textToAnalize_03: Paraghrap 1
nltk_eng_phrases_tokens = word_tokenize(eng_textToAnalize_03_paragraph_1)
print(nltk_eng_phrases_tokens)

print("*) Lemmatization [POS Paraghraph 1]: 1. Tokens printed and formatted: ")
for i in range(len(nltk_eng_phrases_tokens)):
    #print("Token [{}]: {}".format(i + 1, nltk_eng_tokens[i]))
    print("(\""+nltk_eng_phrases_tokens[i]+"\",\"\")," )

###
'''
nltk.download('averaged_perceptron_tagger')
'''
#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()

# function to convert nltk tag to wordnet tag
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

print("*** LEMMATIZE SENTENCE - Paragraph 1 ***")
print(lemmatize_sentence(eng_textToAnalize_03_paragraph_1))
#
print("*** LEMMATIZE SENTENCE - Paragraph 2 ***")
eng_textToAnalize_03_paragraph_2 = "I loved to be going around with my friends, watching the wild nature."
print(lemmatize_sentence(eng_textToAnalize_03_paragraph_2))
#
print("*** LEMMATIZE SENTENCE - Paragraph 3 ***")
eng_textToAnalize_03_paragraph_3 = "I've watched a lot of things, visiting and watching a lot of places, cooking on the roads."
print(lemmatize_sentence(eng_textToAnalize_03_paragraph_3))
#
print("*** LEMMATIZE SENTENCE - Paragraph 4 ***")
eng_textToAnalize_03_paragraph_4 = "We also have runned a lot of dangers expecially the night on desert places with only the breeze to keep us company."
print(lemmatize_sentence(eng_textToAnalize_03_paragraph_4))
#
print("*** LEMMATIZE SENTENCE - Paragraph 5 ***")
eng_textToAnalize_03_paragraph_5 = "An ice cold beer, a good old blues song and the dreams become true."
print(lemmatize_sentence(eng_textToAnalize_03_paragraph_5))

# SINGLE PHRASE STEM
print("## SINGLE PHRASE MANAGEMENT ##")
print("*) Token, Stopwords management, Stemming and lemmatization a single phrase")
text = "We love studying machine learning. We have learned a lot of things."
print("*) Original Phrase")
print(text)
nltk_eng_phrases_tokens = word_tokenize(text)
print("*) Tokens")
print(nltk_eng_phrases_tokens)
print("\tTokens size: " + str(len(nltk_eng_phrases_tokens)))
eng_tokens = word_tokenize(text)
eng_snowball_stemmer = SnowballStemmer("english")
eng_tokes_filtered = []
eng_tokes_removed_filtered = []
print("*) Stopwords removing")
for eng_token in nltk_eng_phrases_tokens:
    if (eng_token.lower() not in nltk_eng_stopwords):
        eng_tokes_filtered.append(eng_token)
    else:
        eng_tokes_removed_filtered.append(eng_token)

print('\t*) ENG PHRASE - NO STOPWORDS')
print("\t"+str(eng_tokes_filtered))
print("\tPhrase no stopword size: " + str(len(eng_tokes_filtered)))
print('\t*) ENG PHRASE - STOPWORDS REMOVED')
print("\t"+str(eng_tokes_removed_filtered) )
print("\tStopword removed size: " + str(len(eng_tokes_removed_filtered)))

# Stem
print("*) Stemming")
eng_tokens_snowball_stem = []
print("TOKEN\t\tSTEM")
for eng_token in eng_tokens:
    eng_token_stem_radix = eng_snowball_stemmer.stem(eng_token)
    eng_tokens_snowball_stem.append(eng_token_stem_radix)
    print("%s\t\t%s" % (eng_token, eng_token_stem_radix) )
print("\tStemming size: " + str(len(eng_tokens_snowball_stem)))

print("*) Lemmatization")
print(lemmatize_sentence(text))

##
### USING OF SPACY LIBRARY ###
print("**********************")
print("** SPACY LIBRARY")
print("**********************")
'''
Spacy library
*) for industrial purpose
*) use only lemmatization process

Af for nltk, we need to download the library itself.
After we need to install the additionals packages to manage 
'''
import spacy
#
print("*) Original phrase")
print(eng_textToAnalize_03_paragraph_1)
spacy_en_nlp = spacy.load("en_core_web_sm")
print("*) Spacy type")
print(type(spacy_en_nlp))
#processing
spacy_processing_eng03_paragraph_1 = spacy_en_nlp(eng_textToAnalize_03_paragraph_1)
print("*) Spacy processing output object type")
print(type(spacy_processing_eng03_paragraph_1))
# token inspection
print("*) Spacy : TOKEN INSPECTION")
print("Token [0]: " + str(spacy_processing_eng03_paragraph_1[0]) )
print("Token [1]:" + str(spacy_processing_eng03_paragraph_1[1]) )
print("Token [2]:" + str(spacy_processing_eng03_paragraph_1[2]) )
print("Token [3]:" + str(spacy_processing_eng03_paragraph_1[3].text) )
print("Token [4]:" + str(spacy_processing_eng03_paragraph_1[4].text) )

print("*) Spacy : TOKEN Type")
print(type(spacy_processing_eng03_paragraph_1[0]))

print("*) Spacy : TOKENS lengths")
print(str(len(spacy_processing_eng03_paragraph_1)) )

print("*) Spacy : PRINT ALL TOKENS")
for token in spacy_processing_eng03_paragraph_1:
    print(token.text)

print("*) Spacy : PRINT ALL TOKENS with index")
for i in range(len(spacy_processing_eng03_paragraph_1)):
    print("Token [{}]: {}".format(i + 1, spacy_processing_eng03_paragraph_1[i]))

'''
Spacey sents
-) sents type of spacy is a "generator" type, so we can't directly access to it's elements. 
-) To access to it's element we can convert esnt to lis object
'''
print("*) Spacy : PRINT ALL sentences with index")
sentences = spacy_processing_eng03_paragraph_1.sents
print("*) Spacy sents type:")
print(type(sentences))

for i,sents in enumerate(sentences):
    print("Token [{}]: {}".format(i + 1, sents))

print("*) Spacy : PRINT ALL sentences with index - [COMPLETE PHRASES WITH PARAGRAPH]")
spacy_processing_eng03 = spacy_en_nlp(eng_textToAnalize_03)
sentences = spacy_processing_eng03.sents
for i,sents in enumerate(sentences):
    print("Token [{}]: {}".format(i + 1, sents))

# [?]
print("*) Spacey - convert sents to list object")
sents_as_list = []
for i,sents in enumerate(sentences):
    sents_as_list.append(sents)
    print("Token [{}]: {}".format(i + 1, sents))


print(type(sents_as_list))
print(str(len(sents_as_list) ) )

for i in range(len(sents_as_list)):
    print("Sentences [{}]: {}".format(i + 1, sents_as_list[i]))

# Spacy lemmatization
print("*) Spacy lemmatization")

eng_tokens_lemmas = []
print("TOKEN\t\tLEMMA")
for spacy_eng_token in spacy_processing_eng03:
    print("%s\t\t%s" % (spacy_eng_token.text, spacy_eng_token.lemma_) )
'''

*) pronouns not have a lemma, spacey indicate that with the 
    appropriate tag: -PRON-

IMPO: REMEMEBR TO USE spacy_eng_token.lemma_ , WITH UNDERSCORE, otherwise spacy
will print the number that represent the hash value of lemma.
'''
print("*) Spacy lemmatization - HASH VALUE")

eng_tokens_lemmas = []
print("TOKEN\t\tLEMMA HASH")
for spacy_eng_token in spacy_processing_eng03:
    print("%s\t\t%s" % (spacy_eng_token.text, spacy_eng_token.lemma) )

# STOPWORDS [ENG]
print("*) Spacy stopwords")
spacy_eng_stopwords_model = spacy_en_nlp.Defaults.stop_words
print(" Spacy stopwords object type")
print(type(spacy_eng_stopwords_model))

print("*) Spacy stopwords length used for ENG")
print(str(len(spacy_eng_stopwords_model)) )

print("*) Spacy stopwords print")
result = list(spacy_eng_stopwords_model)
for i,current_stopword in enumerate(result):
    print("Token [{}]: {}".format(i + 1, current_stopword))

print("*) Spacy stopwords 10 elements")
print(result[:10])

print("TOKEN\t\tLEMMA\t\tIS_STOPWORDS")
for spacy_eng_token in spacy_processing_eng03:
    print("%s\t\t%s\t\t%s" % (spacy_eng_token.text, spacy_eng_token.lemma_,str(spacy_eng_token.is_stop)) )

# Spacy - add all stopwords of phrase to a list
spacy_paraghrap_list = []
spacy_paraghrap_list_is_stop = []
print("TOKEN\t\tLEMMA\t\tIS_STOPWORDS")
for spacy_eng_token in spacy_processing_eng03:
    if(not spacy_eng_token.is_stop):
        spacy_paraghrap_list.append(spacy_eng_token)
    else:
        spacy_paraghrap_list_is_stop.append(spacy_eng_token)

print("*) Spacy NO STOPWORDS 10 elements")
print(spacy_paraghrap_list[:10])
print(type(spacy_paraghrap_list[0]))
print("*) Spacy STOPWORDS 10 elements")
print(spacy_paraghrap_list_is_stop[:10])



###
print("**********************")
print("**********************")

print("**********************")
print("** STEMMING * ITA SECTION *")
print("**********************")

print("**********************")
print("** STEMMING - ITALIAN LANGUAGE")
print("**********************")

print("*** ITA ***")
ita_textToAnalize_02 = "Mi è piaciuto visitare gli Stati Uniti, in particolare correre con una moto sulla Route 66 per " \
                       "fare un classico viaggio su strada. Mi è piaciuto andare in giro con i miei amici, a guardare " \
                       "la natura selvaggia. " \
                       "Ho visto molte cose, visitando e guardando molti posti, cucinando per strada. " \
                       "Abbiamo anche corso molti pericoli, specialmente la notte in luoghi deserti con solo " \
                       "la brezza per farci compagnia. " \
                       "Una birra ghiacciata, una buona vecchia canzone blues e i sogni diventano realtà."
print(ita_textToAnalize_02)

print("*** ITA - Snowball Stemmer algorithm ***")
print(ita_textToAnalize_02)
ita_tokens = word_tokenize(ita_textToAnalize_02)
ita_snowball_stemmer = SnowballStemmer("italian")

word_to_stem = "visitando"
stem_out = ita_snowball_stemmer.stem(word_to_stem)
print("*) Stemming: output for word: " + word_to_stem)
print(stem_out)

word_to_stem = "guardando"
stem_out = ita_snowball_stemmer.stem(word_to_stem)
print("*) Stemming: output for word: " + word_to_stem)
print(stem_out)

#Process snowball stem analisys
ita_tokens_snowball_stem = []
print("TOKEN\t\tSTEM")
for ita_token in ita_tokens:
    ita_token_stem_radix = ita_snowball_stemmer.stem(ita_token)
    ita_tokens_snowball_stem.append(ita_token_stem_radix)
    print("%s\t\t%s" % (ita_token, ita_token_stem_radix) )

print("**********************")
print("** LEMMATIZATION ")
print("**********************")
print("**********************")
print("** LEMMATIZATION - ITA LANGUAGE ")
print("**********************")

print("**********************")
print("** SPACY - ITA LANGUAGE ")
print("**********************")
'''

'''
print("*) Original phrase")
print(ita_textToAnalize_02)
spacy_it_nlp = spacy.load("it_core_news_sm")

print("*) Spacy [IT] : PRINT ALL sentences with index - [COMPLETE PHRASES WITH PARAGRAPH]")
spacy_processing_it_02 = spacy_it_nlp(ita_textToAnalize_02)
sentences = spacy_processing_it_02.sents
for i,sents in enumerate(sentences):
    print("Token [{}]: {}".format(i + 1, sents))

# Spacy lemmatization
print("*) Spacy [IT] lemmatization")

it_tokens_lemmas = []
print("TOKEN\t\tLEMMA")
for spacy_it_token in spacy_processing_it_02:
    print("%s\t\t%s" % (spacy_it_token.text, spacy_it_token.lemma_) )

# STOPWORDS [ITA]
print("*) Spacy [IT] stopwords")
spacy_it_stopwords_model = spacy_it_nlp.Defaults.stop_words
print(" Spacy [IT] stopwords object type")
print(type(spacy_it_stopwords_model))

print("*) Spacy [IT] stopwords length used for ENG")
print(str(len(spacy_it_stopwords_model)) )

print("*) Spacy [IT] stopwords print")
result = list(spacy_it_stopwords_model)
for i,current_stopword in enumerate(result):
    print("Token [{}]: {}".format(i + 1, current_stopword))

print("*) Spacy [IT] stopwords 10 elements")
print(result[:10])

print("TOKEN\t\tLEMMA\t\tIS_STOPWORDS")
for spacy_it_token in spacy_processing_it_02:
    print("%s\t\t%s\t\t%s" % (spacy_it_token.text, spacy_it_token.lemma_,str(spacy_it_token.is_stop)) )

# Spacy - add all stopwords of phrase to a list
spacy_paraghrap_list = []
spacy_paraghrap_list_is_stop = []
print("TOKEN\t\tLEMMA\t\tIS_STOPWORDS")
for spacy_it_token in spacy_processing_it_02:
    if(not spacy_it_token.is_stop):
        spacy_paraghrap_list.append(spacy_it_token)
    else:
        spacy_paraghrap_list_is_stop.append(spacy_it_token)

print("*) Spacy [IT] NO STOPWORDS 10 elements")
print(spacy_paraghrap_list[:10])
print(type(spacy_paraghrap_list[0]))
print("*) Spacy [IT] STOPWORDS 10 elements")
print(spacy_paraghrap_list_is_stop[:10])