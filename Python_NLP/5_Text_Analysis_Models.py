import re

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
vocab_out = build_vocabulary_single(reg_ex_result)
print(vocab_out)