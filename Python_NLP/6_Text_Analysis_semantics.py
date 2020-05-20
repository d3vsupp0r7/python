####

## Section 1: Part of Speech Tagging - (POS)
'''
Part of Speech Tagging - (POS)

Part of Speech Tagging - (POS)  is a process to identify the part of speech of every words
inside a text.
The POS taggging is a low level operation that is useful to create/manage a high level operation, as example
Lemmatization, Tokenization, NAmed entity recognition.

*) POS are:
Adjective    (Aggettivi)    - ADJ
Propositions (Proposizioni) - ADP
Adverbs      (Avverbi)      - ADV
Conjunctions (Congiunzioni) - CONJ
Names        (Nomi)         - NOUN
Numbers      (Numeri)       - NUM
Particle     (Particelle)   - PRT
Pronouns     (Pronomi)      - PRON
Verbs        (Verbi)        - VERB
Punctuation  (Puntegiatura) - .
Other        (Altro)         - X
Definitive articles (Articoli determinativi) - DET

*) POS Techniques

-) Lessical methods
Using a labeled corpus of text, we assign the POS more frequent to a word.

-) Rules based
Analizing the words based on it's form

-) Probabilistics methods
Use some statistical method to determinate the POS Tagging.
Some examples are:
    1) Conditional Random Fields (CRF)
    2) Hidden MarkovModels (HMM)

-) Deep learning
Using Deep Learning neural network to make the POS tagging of a text.
'''
from encodings.utf_8 import encode

from spacy.tokens.span import Span

eng_textToAnalize_01 = "I'd loved to visit U.S.A, in particular running with a motorcycle on Route 66 making a classic" \
                       " trip on the road. " \
                       "I loved to be going around with my friends, watching the wild nature. " \
                       "I've watched a lot of things, visiting and watching a lot of places, cooking on the roads. " \
                       "We also have runned a lot of dangers expecially the night on desert places with only the breeze " \
                       "to keep us company. " \
                       "An ice cold beer, a good old blues song and the dreams become true."

eng_corpus = [
    "I'd loved to visit U.S.A in particular running with a motorcycle on Route 66 making a classic trip on the road.",
    "I loved to be going around with my friends watching the wild nature.",
    "I've watched a lot of things visiting and watching a lot of places cooking on the roads.!",
    "We also have runned a lot of dangers expecially the night on desert places with only the breeze to keep us company.",
    "An ice cold beer a good old blues song and the dreams become true."
]

import nltk

## Download
'''
NTLK have library for managing deep learning text analisis. In order to use nltk for POS tagging, we need to download
the appropriate package:

nltk.download('averaged_perceptron_tagger')

** THIS is one-time operation
'''
# nltk.download('averaged_perceptron_tagger')
#1. Obtain tokens
nltk_eng_tokens = nltk.word_tokenize(eng_textToAnalize_01)
print("*) TOKENS")
print(str(len(nltk_eng_tokens)))
print("*) Tokens type")
print(type(nltk_eng_tokens))
print("TOKEN")

for i in range(len(nltk_eng_tokens)):
    print("Token [{}]: {}".format(i + 1, nltk_eng_tokens[i]))

#
print("# Using nltk to get POS tag from tokens")
nltk_eng_pos_tag = nltk.pos_tag(nltk_eng_tokens)
print("*) TOKENS")
print(str(len(nltk_eng_pos_tag)))
print("*) Tokens with POS type")
print(type(nltk_eng_pos_tag))

for i in range(len(nltk_eng_pos_tag)):
    print("Token with POS TAG [{}]: {}".format(i + 1, nltk_eng_pos_tag[i]))
'''
In order to know tags used for POS TAG in nltk library we need to download the additional package:
nltk.download('tagsets')

** This is a one-time operation
'''
# nltk.download('tagsets')
'''
Print all tags info: nltk.help.upenn_tagset()
'''
print("## NLTK - Some POS Tagging with examples")
#noun
nltk.help.upenn_tagset("NNS")
#Adverbs
nltk.help.upenn_tagset("RB")
# Verbs
nltk.help.upenn_tagset("VBG")
nltk.help.upenn_tagset("VBP")
nltk.help.upenn_tagset("VBN")
nltk.help.upenn_tagset("MD")
##
print("*) Get tokens with it's related nltk pos tagging attribute")
nltk_token_with_tag_string = [token +"("+tag+")" for token, tag in nltk_eng_pos_tag]
print(nltk_token_with_tag_string)
# Build the complete string (token-attribute)
nltk_token_with_tag_string_join = " ".join(nltk_token_with_tag_string)
print(nltk_token_with_tag_string_join)
##############
## SPACY POS TAGGING
#############
import spacy
##############
## SPACY POS TAGGING - ENG
#############
print("## SPACY POS TAGGING - ENG")
print("*) Original phrase")
print(eng_textToAnalize_01)

#Spacy model loading e text processing
eng_spacy_model = spacy.load("en_core_web_sm")
eng_spacy_doc = eng_spacy_model(eng_textToAnalize_01)

print("*) Spacy simple tag output")
print("Spacy Token Text: " + eng_spacy_doc[0].text)
print("Spacy POS TAG ( tag_ attribute)  : " + eng_spacy_doc[0].tag_)
print("Spacy POS TAG ( pos_ attribute)  : " + eng_spacy_doc[0].pos_)
'''
IMPO: Remember to use underscore when we wont to print the
 POS tagging with spacy:
 .tag_ : we get acronym
 .pos_ : we get explicative
'''
# Get spacy pos tag explanation
print("*) Get spacy tag explanation")
print(spacy.explain(eng_spacy_doc[0].tag_))

###### ?
#for token in eng_spacy_doc:
    #print(token+"\t\t"+token.pos_+"\t\t"+token.tag_+"\t\t"+spacy.explain(token.tag_))
    #print(token[0]+"\t\t"+token.pos_+"\t\t"+token.tag_+"\t\t")
    #print(token[0])
##### ?

print("Token_NBR"+"\t\t"+"Token"+"\t\t"+"Token POS"+"\t\t"+"Token TAG"+"\t\t"+"Token Explanation")
for i in range(len(eng_spacy_doc)):
    print("Token [{}]: {} - {} - {} - {}".format(i + 1, eng_spacy_doc[i], eng_spacy_doc[i].pos_,eng_spacy_doc[i].tag_, spacy.explain(eng_spacy_doc[i].tag_)) )

print("## Named Entity Recognition - NER")
'''
Named Entity Recognition - NER

Named Entity Recognition - NER is a process for identify membership class
of a word inside a text document.

This mean how to classify a word into a category, such as
*) Persons names,
*) Organizations,
*) Locations,
*) Quantity (numeric/textual)
*) Money (numeric/textual)
*) Dates

IMPORTANT NOTE: NLTK not provide NER function for analisys. So we use spacy

'''
ner_eng_phrase_01 = "Amazon founder Jeff Bezos purchased the Washinton Post for $ 520 million in October 2013."
'''
NER analysis on text:

    Amazon [ORGANIZATION] founder 
    Jeff Bezos [PERSON] 
    purchased the 
    Washinton Post [ORGANIZATION] for 
    $ 520 million in 
    October 2013. [DATE]
'''
ner_eng_spacy_doc = eng_spacy_model(ner_eng_phrase_01)
print("*) Spacy ents type")
print(type(ner_eng_spacy_doc.ents))
print("*) Spacy ents length")
print(str(len(ner_eng_spacy_doc.ents)) )

print("*) Spacy NER output")
print(ner_eng_spacy_doc.ents)

print("*) Spacy NER access")
print(ner_eng_spacy_doc.ents[0])
print(ner_eng_spacy_doc.ents[3])
print(ner_eng_spacy_doc.ents[4])

print("*) Spacy NER: Get NER CATEGORY")
print(type( ner_eng_spacy_doc.ents[0].label_))
print( ner_eng_spacy_doc.ents[0].label_)

print("Entity recognized: " + str(ner_eng_spacy_doc.ents[0]) + " - Category: " + str(ner_eng_spacy_doc.ents[0].label_) )
print("Entity recognized: " + str(ner_eng_spacy_doc.ents[1]) + " - Category: " + str(ner_eng_spacy_doc.ents[1].label_) )
print("Entity recognized: " + str(ner_eng_spacy_doc.ents[2]) + " - Category: " + str(ner_eng_spacy_doc.ents[2].label_) )
print("Entity recognized: " + str(ner_eng_spacy_doc.ents[3]) + " - Category: " + str(ner_eng_spacy_doc.ents[3].label_) )
print("Entity recognized: " + str(ner_eng_spacy_doc.ents[4]) + " - Category: " + str(ner_eng_spacy_doc.ents[4].label_) )
#print(ner_eng_spacy_doc.ents[1] + " Category: " + ner_eng_spacy_doc.ents[1].label_)
#print(ner_eng_spacy_doc.ents[2] + " Category: " + ner_eng_spacy_doc.ents[2].label_)
#print(ner_eng_spacy_doc.ents[3] + " Category: " + ner_eng_spacy_doc.ents[3].label_)
#print(ner_eng_spacy_doc.ents[4] + " Category: " + ner_eng_spacy_doc.ents[4].label_)

print("*) Spacy NER: Get NER CATEGORY Explanation")
print("\tEntity recognized: " + str(ner_eng_spacy_doc.ents[0]) )
print("\t\t"+spacy.explain(ner_eng_spacy_doc.ents[0].label_) )

print("\tEntity recognized: " + str(ner_eng_spacy_doc.ents[1]) )
print("\t\t"+spacy.explain(ner_eng_spacy_doc.ents[1].label_) )

print("\tEntity recognized: " + str(ner_eng_spacy_doc.ents[2]) )
print("\t\t"+spacy.explain(ner_eng_spacy_doc.ents[2].label_) )

print("\tEntity recognized: " + str(ner_eng_spacy_doc.ents[3]) )
print("\t\t"+spacy.explain(ner_eng_spacy_doc.ents[3].label_) )

print("\tEntity recognized: " + str(ner_eng_spacy_doc.ents[4]) )
print("\t\t"+spacy.explain(ner_eng_spacy_doc.ents[4].label_) )

# Cycling on spacy entity
print("*) Cycling on spacy entity")
print("Entity\t\tTAG\t\tDESCRIPTION")
for ent in ner_eng_spacy_doc.ents:
    print(ent.text+"\t\t"+ent.label_+"\t\t"+spacy.explain(ent.label_))

ner_eng_phrase_01_ner_formatted = ner_eng_phrase_01
for i in range(0,len(ner_eng_spacy_doc.ents)):
    ner_eng_phrase_01_ner_formatted = ner_eng_phrase_01_ner_formatted.replace(ner_eng_spacy_doc.ents[i].text, ner_eng_spacy_doc.ents[i].text+"("+ner_eng_spacy_doc.ents[i].label_+")")

print("*) PRINT Spacy NER Formatted string")
print(ner_eng_phrase_01_ner_formatted)

#######################
#######################
#######################

##############
## SPACY POS TAGGING - ITA
#############
print("## SPACY POS TAGGING - ITA")
ita_textToAnalize_01 = "Mi è piaciuto visitare gli Stati Uniti, in particolare correre con una moto sulla Route 66 per " \
                       "fare un classico viaggio su strada. Mi è piaciuto andare in giro con i miei amici, a guardare " \
                       "la natura selvaggia. " \
                       "Ho visto molte cose, visitando e guardando molti posti, cucinando per strada. " \
                       "Abbiamo anche corso molti pericoli, specialmente la notte in luoghi deserti con solo " \
                       "la brezza per farci compagnia. " \
                       "Una birra ghiacciata, una buona vecchia canzone blues e i sogni diventano realtà."
print("*) Original phrase")
print(ita_textToAnalize_01)

#Spacy model loading e text processing
ita_spacy_model = spacy.load("it_core_news_sm")
ita_spacy_doc = ita_spacy_model(ita_textToAnalize_01)

'''
For the italian language model, the spacy library not provide the explanation of tag.

the .tag_ contain additional info

'''

print("## SPACY - ITA POS TAGGING OUTPUT")
print("Token_NBR"+"\t\t"+"Token"+"\t\t"+"Token POS"+"\t\t"+"Token TAG"+"\t\t"+"Token Explanation")
for i in range(len(ita_spacy_doc)):
    print("Token [{}]: {} - {} - {} - {}".format(i + 1, ita_spacy_doc[i], ita_spacy_doc[i].pos_,ita_spacy_doc[i].tag_, spacy.explain(ita_spacy_doc[i].tag_)) )

#
ner_ita_phrase_01 = "Jeff Bezos, il fondatore di Amazon, ha acquistato il Washinton Post per 520 milioni di dollari nell'Ottobre 2013."
'''
NER analysis on text:
  Jeff Bezos [PERSON], 
  il fondatore di 
  Amazon [ORGANIZATION],
  ha acquistato il 
  Washinton Post [ORGANIZATION] per 
  250 milioni di dollari [MONEY]
  nell'Ottobre 2013. [DATE]"
'''
# Ciclinc on spacy entity
ner_ita_spacy_doc = ita_spacy_model(ner_ita_phrase_01)

print("*) Cycling on spacy entity")
print("Entity\t\tTAG\t\tDESCRIPTION")
for ent in ner_ita_spacy_doc.ents:
    print(ent.text+"\t\t"+ent.label_+"\t\t"+spacy.explain(ent.label_))

ner_ita_phrase_01_ner_formatted = ner_ita_phrase_01
for i in range(0,len(ner_ita_spacy_doc.ents)):
    ner_ita_phrase_01_ner_formatted = ner_ita_phrase_01_ner_formatted.replace(ner_ita_spacy_doc.ents[i].text, ner_ita_spacy_doc.ents[i].text+"("+ner_ita_spacy_doc.ents[i].label_+")")

print("*) PRINT Spacy NER Formatted string - 01")
print(ner_ita_phrase_01_ner_formatted)
ner_ita_phrase_01_ner_formatted_first = ner_ita_phrase_01_ner_formatted

##### ? -> Correct
ner_ita_phrase_02 = "Jeff Bezos, il fondatore di Amazon, ha acquistato il Washinton Post per 520 milioni di euro nell'Ottobre 2013."
#ner_ita_spacy_doc = ita_spacy_model(ner_ita_phrase_02)
#ner_ita_phrase_01_ner_formatted = ner_ita_phrase_02
#print("*) PRINT Spacy NER Formatted string - 02")
#for i in range(0,len(ner_ita_spacy_doc.ents)):
#    ner_ita_phrase_01_ner_formatted = ner_ita_phrase_01_ner_formatted.replace(ner_ita_spacy_doc.ents[i].text, ner_ita_spacy_doc.ents[i].text+"("+ner_ita_spacy_doc.ents[i].label_+")")

print("*) Spacy ENTITY CORRECTION - ITA")
'''
Obtaing hash of labels with spacy

'''
money_tag = ner_ita_spacy_doc.vocab.strings["MONEY"]
print(money_tag)
print("*) Original phrase")
print(ner_ita_phrase_01)
print(ner_ita_spacy_doc.ents)
## IMPO: SPACY DEFINE AN ADHOC NER ENTITY - MONEY
it_money_ner = Span(ner_ita_spacy_doc,14,18,label=money_tag)
print(it_money_ner)

ner_ita_spacy_doc.ents = list(ner_ita_spacy_doc.ents)+[it_money_ner]
print(ner_ita_spacy_doc.ents)
ner_ita_phrase_01_ner_formatted = ner_ita_phrase_01
#for i in range(0,len(ner_ita_spacy_doc.ents)):
#    ner_ita_phrase_01_ner_formatted = ner_ita_phrase_01_ner_formatted.replace(ner_ita_spacy_doc.ents[i].text, ner_ita_spacy_doc.ents[i].text+"("+ner_ita_spacy_doc.ents[i].label_+")")
ner_ita_phrase_01_ner_formatted = ner_ita_phrase_01
for i in range(0,len(ner_ita_spacy_doc.ents)):
    ner_ita_phrase_01_ner_formatted = ner_ita_phrase_01_ner_formatted.replace(ner_ita_spacy_doc.ents[i].text, ner_ita_spacy_doc.ents[i].text+"("+ner_ita_spacy_doc.ents[i].label_+")")

print("*) PRINT Spacy NER Formatted string - 01 - + new ENTITY - MONEY")
print(ner_ita_phrase_01_ner_formatted)
## IMPO: SPACY DEFINE AN ADHOC NER ENTITY - DATE
## New Entity preparation
print("*) PRINT Spacy NER Formatted string - 01 - + new ENTITY - DATE")
date_tag = ner_ita_spacy_doc.vocab.strings["DATE"]
print(date_tag)
it_date_ner = Span(ner_ita_spacy_doc,18,20,label=date_tag)
print(it_date_ner)
##
ner_ita_spacy_doc.ents = list(ner_ita_spacy_doc.ents)+[it_date_ner]
print(ner_ita_spacy_doc.ents)
ner_ita_phrase_01_ner_formatted = ner_ita_phrase_01
for i in range(0,len(ner_ita_spacy_doc.ents)):
    ner_ita_phrase_01_ner_formatted = ner_ita_phrase_01_ner_formatted.replace(ner_ita_spacy_doc.ents[i].text, ner_ita_spacy_doc.ents[i].text+"("+ner_ita_spacy_doc.ents[i].label_+")")
print("*) PRINT Spacy NER Formatted string - 01 - + new ENTITY - DATE")
print(ner_ita_phrase_01_ner_formatted)
'''
Possible problems:
*) If we get exception on change the existing entity, ee need to remove first the invalid one.
'''
# step1
ents = list(ner_ita_spacy_doc.ents)
print(ents)
# step 2
del(ents[0])
print(ents)
# step 3
ner_ita_spacy_doc.ents = ents + [it_date_ner]
print(ner_ita_spacy_doc.ents)
#step 4
for i in range(0,len(ner_ita_spacy_doc.ents)):
    ner_ita_phrase_01_ner_formatted = ner_ita_phrase_01_ner_formatted.replace(ner_ita_spacy_doc.ents[i].text, ner_ita_spacy_doc.ents[i].text+"("+ner_ita_spacy_doc.ents[i].label_+")")
print(ner_ita_phrase_01_ner_formatted_first)
print(ner_ita_phrase_01_ner_formatted)

## Spacy: module Displacy
from spacy import displacy
print(displacy.render(ner_ita_spacy_doc, style="ent") )
displacy.render(ner_ita_spacy_doc, style="ent")
## TO SHOW colored entity texts
## http://localhost:5000/
##displacy.serve(ner_ita_spacy_doc, style="ent")
#from pathlib import Path
#svg = displacy.render(ner_ita_spacy_doc, style="dep", jupyter=False)
#file_name = '-'.join([w.text for w in ner_ita_spacy_doc if not w.is_punct]) + ".svg"
#output_path = Path("C:/pythonGithub/python/Python_NLP/" + file_name)
#output_path.open("w", encoding="utf-8").write(svg)