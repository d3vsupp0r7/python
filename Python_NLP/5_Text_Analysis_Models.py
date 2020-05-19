'''
Text encoding

'''

'''
Algorithms

# Bag of Words

This algorithms is the simplier, manage text as document of common lenght and inside this documents
text the arrangement of words have no importance.
Another particular of bag of words model, is that is associate words and their index,
and this relationship is bidirectional.

'''
import numpy as np
import re
from nltk import word_tokenize

eng_textToAnalize_01 = "I'd loved to visit U.S.A, in particular running with a motorcycle on Route 66 making a classic" \
                       " trip on the road. " \
                       "I loved to be going around with my friends, watching the wild nature. " \
                       "I've watched a lot of things, visiting and watching a lot of places, cooking on the roads. " \
                       "We also have runned a lot of dangers expecially the night on desert places with only the breeze " \
                       "to keep us company. " \
                       "An ice cold beer, a good old blues song and the dreams become true."
#

print('*) Example corpus_tokenizer - Tokenize list type contains lists itself')
def corpus_tokenizer(text):
    tokens = [re.split("\\W+", sent.lower()) for sent in text]
    return tokens

string_sentences = ["This is phrase 1.", "This also phrase 2", "Again! This is phrase 3"]
print("*)Original phrase")
print(string_sentences)
print(type(string_sentences))
tokens_result = corpus_tokenizer(string_sentences)
print("*)Tokenized  phrase result")
print(tokens_result)

### declare vocabulary definition function
def build_vocabulary(text):
    vocab = set({})
    for tokens in text:
        for token in tokens:
            vocab.add(token)
    return list(vocab)

print("*) VOCABS definition - list of lists")
print("\t*)Original phrase")
print(string_sentences)
print("\t*)Vocabulary result")
vocab_out = build_vocabulary(tokens_result)
print(vocab_out)
#
string_sentences = ["This is phrase one.", "This also phrase two", "Again! This is phrase three"]
print("\t*)Original phrase")
print(string_sentences)
print("\t*)Vocabulary result")
tokens_result = corpus_tokenizer(string_sentences)
vocab_out = build_vocabulary(tokens_result)
print(vocab_out)

##
print('*) Example corpus tokenizer - paragraph')
reg_ex_result = re.split("\W+", eng_textToAnalize_01)
print(type(reg_ex_result))
print("\t*) Tokens from paragraph result")
print(reg_ex_result)

### declare vocabulary definition function -sigle string
def build_vocabulary_single(text):
    vocab = set({})
    for token in text:
            vocab.add(token)
    return list(vocab)

print("\t*) Vocabulary result")
'''
This also mapping index->word relation of bag of words model.
Marked this relation in comment: [*] vocab_index_to_word

'''
vocab_out = build_vocabulary_single(reg_ex_result)
print(vocab_out)
print("*) vocab length")
print(str(len(vocab_out)))
vocab_out_len = len(vocab_out)
'''
[*] vocab_index_to_word
vocab_index_to_word = build_vocabulary_single(reg_ex_result)
print(vocab_index_to_word)
'''

## BAG OF WORDS - 1. dictionary
word_to_index = dict([word, i] for i, word in enumerate(vocab_out))
print(word_to_index)
print('*) word_to_index length:')
print(str(len(word_to_index)))
print('*) Getting index of a word:')
print(word_to_index["blues"])
print(word_to_index["song"])
print(word_to_index["beer"])

print('*) Getting word of a index:')
print(vocab_out[10])
print(vocab_out[50])
print(vocab_out[64])
'''
Index is out of bound, this line throw and exception
print(vocab_out[75])
'''

### BAG OF WORD - uses of numpy for rapresentation
# 1. use nltk to take senetence representation. this will represent the phrases

from nltk.tokenize import sent_tokenize
sentences_tokens = sent_tokenize(eng_textToAnalize_01)
print(sentences_tokens)
print(type(sentences_tokens) )
print(str(len(sentences_tokens) ) )
n_phrases = len(sentences_tokens)

for i in range(len(sentences_tokens)):
    print("Token [{}]: {}".format(i + 1, sentences_tokens[i]))

################
################
################
corpus = [
    "I'd loved to visit U.S.A in particular running with a motorcycle on Route 66 making a classic trip on the road.",
    "I loved to be going around with my friends watching the wild nature.",
    "I've watched a lot of things visiting and watching a lot of places cooking on the roads.!",
    "We also have runned a lot of dangers expecially the night on desert places with only the breeze to keep us company.",
    "An ice cold beer a good old blues song and the dreams become true."
]
print('** Managing text using lists **')
#1. Sentences into text
corpus_tokens = corpus_tokenizer(corpus)
print(str(len(corpus_tokens)))
print(type(corpus_tokens))
print(corpus_tokens)
#Vocabulary of processed text
vocab_out =  build_vocabulary(corpus_tokens)
print(str(len(vocab_out)))
print(type(vocab_out))
print(vocab_out)
#
print(vocab_out[3])
#
print('*) word to index')
word_to_index = dict([word, i] for i,word in enumerate(vocab_out))
print(word_to_index["beer"])
#
des_count = len(corpus)
print("des_count: " + str(des_count))
vocab_size = len(vocab_out)
print("vocab size: " + str(vocab_size))
# bow
print("*) BOW with numpy")
corpus_bow = np.zeros((des_count, vocab_size))
print(corpus_bow.shape)

for i,tokens in enumerate(corpus_tokens):
    for token in tokens:
        corpus_bow[i][word_to_index[token]] +=1

print(type(corpus_bow) )
print("*) Numpy array output")
print(corpus_bow)

'''
At every execution of code, the dictionary/set position can vary. 
We print the first 10 element of each paragraph to show the differences.
'''
print("*) Numpy array output - first row")
print(corpus_bow[0])
print(type(corpus_bow[0]))
'''
1. 0. 0. 0. 0. 1. 0. 1. 1. 1.
'''

print("*) Numpy array output - second row")
print(corpus_bow[1])
'''
1. 0. 1. 0. 0. 0. 0. 1. 0. 0.
'''

print("*) Numpy array output - third row")
print(corpus_bow[2])
'''
1. 0. 0. 0. 0. 0. 0. 0. 0. 0.
'''
print("*) Numpy array output - fourth row")
print(corpus_bow[3])
'''
1. 1. 0. 1. 0. 0. 1. 1. 0. 0.
'''

print("*) Numpy array output - fifth row")
print(corpus_bow[4])
'''
1. 0. 0. 0. 1. 0. 0. 0. 0. 0.
'''

def bag_of_words(corpus, return_vocab=False):
    corpus_tokens = corpus_tokenizer(corpus)
    index_to_word = build_vocabulary(corpus_tokens)
    word_to_index = dict([word,i] for i,word in enumerate(index_to_word))
    #
    docs_count = len(corpus)
    vocab_size = len(index_to_word)
    #
    corpus_bow = np.zeros((docs_count, vocab_size))
    #
    for i,tokens in enumerate(corpus_tokens):
        for token in tokens:
            corpus_bow[i][word_to_index[token]]+=1

    if(return_vocab):
        return corpus_bow, index_to_word
    else:
        return corpus_bow

corpus_bow, vocab = bag_of_words(corpus,return_vocab=True)
print("** BOW - vocab **")
print(vocab)
print(type(vocab) )
print(str(len(vocab)))

print("** BOW - Analysis sentence with it relative BOW **")
for sent,bow in zip(corpus,corpus_bow):
    print("Sentence: ", sent)
    print("Bag of word: ", bow)
    print("----------")

### BAG OF WORD - END
################
################

################
################
### TF*IDF - START
'''
TF*IDF
Term Frequency * Inverse Document Frequency

In this type of modelling, we have penalty fro common words ang give more weigth to
rare words.

[*] Term Frequency             => measure frequency of every term into document
[*] Inverse Document Frequency => measure the importance ov every term inside the all document.

'''


### TF*IDF - START
################
################
print("### TF*IDF ###")
print("*) original words")
print(corpus)
#1. Determinate the Document frequency
'''
[*] Document frequency: How many documents contains a certain word

IDF = log(num of docs/Document frequency)

[*] Term frequency
To calculate "term frequency" we need to remove stopwords.
Thi is specific for every document.
From the cleaned text, we calculate, for each paragraph the term frequency as follow:

(number of time word appears into document)
------------------------------------------
(number total of wors inside the document)
'''
print("### TF*IDF using numpy")
corpus_tokens = corpus_tokenizer(corpus)
index_to_word = build_vocabulary(corpus_tokens)
word_to_index = dict([word,i] for i,word in enumerate(index_to_word))
docs_count = len(corpus)
vocab_size = len(index_to_word)

#1. Document frequency
df = np.zeros(vocab_size)
print(df.shape)
for i,word in enumerate(index_to_word):
    for tokens in corpus_tokens:
        if(word in tokens):
            df[i]+=1

print(index_to_word)
print(df)

# Inverse document frequency
idf = np.log(docs_count/df)+1
print("*) IDF")
print(idf)
print(type(idf) )
print(str(len(idf) ) )

# Term frequency
tf = np.zeros((docs_count,vocab_size))

for i,tokens in enumerate(corpus_tokens):
    word_counts = len(tokens)
    for token in tokens:
        tf[i][word_to_index[token]]+=1
    tf[i]/=word_counts

print("*) TF*IDF result")
print(tf)

print("*) Final output TF*IDF")
tf_idf = tf*idf
print(tf_idf)

print("## Notes")
print(tf.shape)
print(idf.shape)

def tf_idf(corpus, return_vocab=False):
    corpus_tokens = corpus_tokenizer(corpus)
    index_to_word = build_vocabulary(corpus_tokens)
    word_to_index = dict([word, i] for i, word in enumerate(index_to_word))
    #
    docs_count = len(corpus)
    vocab_size = len(index_to_word)
    # 1. Document frequency
    df = np.zeros(vocab_size)
    for i, word in enumerate(index_to_word):
        for tokens in corpus_tokens:
            if (word in tokens):
                df[i] += 1
    #Inverse Document Frequency
    idf = np.log(docs_count/df)+1
    # Term frequency
    tf = np.zeros((docs_count, vocab_size))

    for i, tokens in enumerate(corpus_tokens):
        word_counts = len(tokens)
        for token in tokens:
            tf[i][word_to_index[token]] += 1
        tf[i] /= word_counts
    #tf*idf
    tf_idf = tf * idf
    if (return_vocab):
        return tf_idf, index_to_word
    else:
        return tf_idf

corpus_tfidf, vocab = tf_idf(corpus, return_vocab=True)
print("*) TF*IDF vocab")
print(vocab)

for sent,tfidf in zip(corpus,corpus_tfidf):
    print("Sentence: ", sent)
    print("TF-IDF: ", tfidf)
    print("----------")
'''

'''
