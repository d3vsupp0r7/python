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

eng_textToAnalize_01 = "I'd loved to visit U.S.A, in particular running with a motorcycle on Route 66 making a classic" \
                       " trip on the road. " \
                       "I loved to be going around with my friends, watching the wild nature. " \
                       "I've watched a lot of things, visiting and watching a lot of places, cooking on the roads. " \
                       "We also have runned a lot of dangers expecially the night on desert places with only the breeze " \
                       "to keep us company. " \
                       "An ice cold beer, a good old blues song and the dreams become true."

corpus = [
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