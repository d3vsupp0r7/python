import nltk
from nltk.corpus import stopwords
from os import listdir
import string

'''
IMPO: 
NLTK is not implement to manage a big dataset.
we can limit to use 1000 sample for class
'''
def get_dataset(files_path, labels=["pos","neg"], samples_per_class=None):
    dataset = []
    for label in labels:
        count = 0
        path = files_path+"/" + label
        print("path search fun: " + path)
        for file in listdir(path):
            review_file = open(path + "/" + file, encoding="utf-8")
            #Manage the open review file
            review = review_file.read()
            review = review.translate(str.maketrans('','',string.punctuation))
            review = review.lower()
            #Token extraction using nltk
            words = nltk.word_tokenize(review)
            #
            #words_filtered = [word for word in words if word not in stopwords.words("english")]
            #words_dict = dict([word,True] for word in words_filtered)
            words_dict = dict([word, True] for word in words if word not in stopwords.words("english"))
            dataset.append((words_dict,label))
            #? -> dataset.append((words_dict,label,file))
            count+=1
            if(samples_per_class!=None):
                if(count>=samples_per_class):
                    break
    return dataset

ds_files_path = "C:/dev/nlp_dataset/nlp_ml_dataset_aclImdb/"
train_set = get_dataset(ds_files_path+"train",samples_per_class=1000)
#train_set = get_dataset(ds_files_path+"train",samples_per_class=500)
print("*) Dataset type:")
print(type(train_set))
print("*) Dataset length:")
print(len(train_set))
print("*) Dataset: print the first value")
print(train_set[0])
'''
Example result:

({'bromwell': True, 'high': True, 'cartoon': True, 'comedy': True, 'ran': True, 'time': True, 'programs': True, 
'school': True, 'life': True, 'teachers': True, '35': True, 'years': True, 'teaching': True, 'profession': True, 
'lead': True, 'believe': True, 'highs': True, 'satire': True, 'much': True, 'closer': True, 'reality': True, 
'scramble': True, 'survive': True, 'financially': True, 'insightful': True, 'students': True, 'see': True, 
'right': True, 'pathetic': True, 'pomp': True, 'pettiness': True, 'whole': True, 'situation': True, 
'remind': True, 'schools': True, 'knew': True, 'saw': True, 'episode': True, 'student': True, 
'repeatedly': True, 'tried': True, 'burn': True, 'immediately': True, 'recalled': True, 
'classic': True, 'line': True, 'inspector': True, 'im': True, 'sack': True, 'one': True, 
'welcome': True, 'expect': True, 'many': True, 'adults': True, 'age': True, 'think': True, 'far': True,
 'fetched': True, 'pity': True, 'isnt': True}, 'pos')
 
 
'''
ds_files_path = "C:/dev/nlp_dataset/nlp_ml_dataset_aclImdb/"
test_set = get_dataset(ds_files_path+"test",samples_per_class=1000)
#test_set = get_dataset(ds_files_path+"test",samples_per_class=500)
print(len(test_set))
print(test_set[0])

## BUILD CLASSIFIER MODEL WITH NLTK
'''
NLTK use a  Naive Bayes model for classify test.
'''
from nltk.classify import NaiveBayesClassifier
nltk_NaiveBayes_classifier = NaiveBayesClassifier.train(train_set)

# Quality check for model
print("## NLTK Quality check for model ##")
from nltk.classify.util import accuracy
'''
accuracy: give percentage of correct classification executed by the model.
'''
accuracy_result_train = accuracy(nltk_NaiveBayes_classifier,train_set)
print("accuracy on train dataset:")
print(accuracy_result_train)
accuracy_result_test = accuracy(nltk_NaiveBayes_classifier,test_set)
print("accuracy on test dataset:")
print(accuracy_result_test)
'''

'''

'''
IMPO:
NLTK give us the chance to check what are the feature (word) most significant for our model.

This feature gibve information with following tag:
 neg : pos    =     17.0 : 1.0
 the weigth is given for tme most value, as example
 
 pointless = True              neg : pos    =     17.0 : 1.0
 
 pointless has most negative importance.
 
'''
print(nltk_NaiveBayes_classifier.show_most_informative_features())

print("## MANAGE A NEW REVIEW")
'''

'''
def create_word_features(review):
    review = review.translate(str.maketrans('', '', string.punctuation))
    review = review.lower()
    words = nltk.word_tokenize(review)
    words_dict = dict([word, True] for word in words if word not in stopwords.words("english"))
    return words_dict
# Define new review
good_review = "This movie was just great"
x_new_phrase = create_word_features(good_review)
result = nltk_NaiveBayes_classifier.classify(x_new_phrase)
print("*) result for new phrase")
print(result)
# Define new review
bad_review = "This movie was just terrible"
x_new_phrase = create_word_features(bad_review)
result = nltk_NaiveBayes_classifier.classify(x_new_phrase)
print("*) result for new phrase")
print(result)
# Define new review
complex_review_example_01 = "This is a good movie. Some choices are very bad and i don't like it. But over all i've appreciate the meanings."
x_new_phrase = create_word_features(complex_review_example_01)
result = nltk_NaiveBayes_classifier.classify(x_new_phrase)
print("*) result for new phrase")
print(result)
#
'''
SINGLE PHRASE RESULT

*) result for new phrase
pos
*) result for new phrase
neg
*) result for new phrase
pos
'''