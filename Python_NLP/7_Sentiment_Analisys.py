####

## Sentiment Analysis
'''
Sentiment Analysis

Sentiment Analisys is a process to identify emotions, positivee or negative,
inside a textual content.

As simple introduction, some word inside a text give a  sentintiment about the entire text.
We can associate to these significant word a weight, that goes from 0, not significant, to 1 (most significant).

To get the entire feeling of text, we need to sum all the weight associate to relevant words.
For negative word, we use a negative weight tha going from 0 for no significant word to -1 indicates a high negative.
As for positive, the negative weight is the sum of all word with negative weight.

A text con also mix positive and negative words. The result of this text is also a sum of element.
we subtract negative word weight and add positive weight.

*) Algorithms for determinate weight of words
-] VADER : Valene Aware Dictionary for Sentiment Reasoning
Use of annoted dictionary with lessical rules

-] Machine Learning model
Using a Machine Learning model, and traing it on a sufficiently large text.


'''

##############
## SENTIMENTAL ANALYSIS
#############

##############
## SENTIMENTAL ANALYSIS - ENG - VADER MODEL
#############
#### NLTK LIBRARY
import nltk
from nltk.sentiment import SentimentAnalyzer

'''
In order to use VADER method for sentiment analisys with nltk library, we need to download
an appropriate package:

nltk.download("vader_lexicon")

This is a one-time operation
'''
#nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SentimentAnalyzer
#1. istantiate class
eng_nltk_sa_class = SentimentAnalyzer()
text = "I love this app."
result = eng_nltk_sa_class.polarity_scores(text)
#Valuate sentiment using
print("*) Original phrase")
print(eng_nltk_sa_class)
print("*) NLTK SentimentAnalyzer - type")
print(type(eng_nltk_sa_class) )
print("*) NLTK SentimentAnalyzer - result")
print(result)
print("*) NLTK SentimentAnalyzer - result type")
print(type(result) )
print("*) NLTK SentimentAnalyzer - result size")
print(str(len(result) ) )
'''
{'neg': 0.0, 'neu': 0.323, 'pos': 0.677, 'compound': 0.6369}
neg: negative words point
neu: neutral words point
pos: positive words point
compound: sentiment general of text: -1 totally negative/ 0 neutral/1 totally positive

Notes:
*) words in uppercase have major weight
*) punctuation influence the weight 
'''
text = "I didn't like this app, uninstalled"
result = eng_nltk_sa_class.polarity_scores(text)
print("*) NLTK SentimentAnalyzer - negative result example")
print(result)
'''
'''

eng_paragraph_01 ="I'd loved to visit U.S.A in particular running with a motorcycle on Route 66 making a classic trip on the road."
eng_paragraph_02 ="I loved to be going around with my friends watching the wild nature."
eng_paragraph_03 ="I've watched a lot of things visiting and watching a lot of places cooking on the roads.!"
eng_paragraph_04 ="We also have runned a lot of dangers expecially the night on desert places with only the breeze to keep us company."
eng_paragraph_05 ="An ice cold beer a good old blues song and the dreams become true."

print("*) NLTK SentimentAnalyzer - eng_paragraph_01 result example")
result = eng_nltk_sa_class.polarity_scores(eng_paragraph_01)
print(result)

print("*) NLTK SentimentAnalyzer - eng_paragraph_02 result example")
result = eng_nltk_sa_class.polarity_scores(eng_paragraph_02)
print(result)

print("*) NLTK SentimentAnalyzer - eng_paragraph_03 result example")
result = eng_nltk_sa_class.polarity_scores(eng_paragraph_03)
print(result)

print("*) NLTK SentimentAnalyzer - eng_paragraph_04 result example")
result = eng_nltk_sa_class.polarity_scores(eng_paragraph_04)
print(result)

print("*) NLTK SentimentAnalyzer - eng_paragraph_05 result example")
result = eng_nltk_sa_class.polarity_scores(eng_paragraph_05)
print(result)

print("*) NLTK SentimentAnalyzer - UPPERCASE TEST result example")
text_01="I love this app, even if user interface is poor"
result = eng_nltk_sa_class.polarity_scores(text_01)
print(result)
print("\t uppercase example")
text_01="I LOVE this app, even if user interface is poor"
result = eng_nltk_sa_class.polarity_scores(text_01)
print(result)
print("\t punctuation example")
text_01="I LOVE!!! this app, even if user interface is poor"
result = eng_nltk_sa_class.polarity_scores(text_01)
print(result)

#############
#############
#############
print("*********************************************")
## Example 2 -
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sentences=["hello","why is it not working?!"]
sid = SIA()
for sentence in sentences:
    ss = sid.polarity_scores(sentence)
    print(ss)

