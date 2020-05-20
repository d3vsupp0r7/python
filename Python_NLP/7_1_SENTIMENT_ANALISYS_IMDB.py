reviews = []
sentiment_result = []
labels = ["pos","neg"]
label_map = {"pos":1,"neg":0}

#
#
from os import listdir
files_path = "C:/dev/nlp_dataset/nlp_ml_dataset_aclImdb/train/"
out = listdir(files_path+"/pos")
print("*) File name list")
print(out)
print("*) File type")
print(type(out))
print("*) Files number in directory")
print(str(len(out)))
#
files_path = "C:/dev/nlp_dataset/nlp_ml_dataset_aclImdb/train"
print(listdir(files_path+"/pos") )
##
for label in labels:
    path = files_path+"/"+label
    for file in listdir(path):
        review_file = open(path+"/"+file,encoding="utf-8")
        #
        review = review_file.read()
        review_file.close()
        #
        reviews.append(review)
        sentiment_result.append(label_map[label])
#
print(reviews[0])
print(reviews[-1])

print(reviews[:3])
print(reviews[-3:])

print(sentiment_result[:10])
print(sentiment_result[-10:])

##
ds_files_path = "C:/dev/nlp_dataset/nlp_ml_dataset_aclImdb/"

def load_sentiments_files(path,labels=["pos","neg"]):
    reviews = []
    sentiment_result = []
    #labels = ["pos","neg"]
    label_map = {"pos":1,"neg":0}

    for label in labels:
        #path = ds_files_path+"/"+label
        path = path +"/"+ label
        print("path search fun: " + path)
        for file in listdir(path):
            review_file = open(path+"/"+file,encoding="utf-8")
            #
            review = review_file.read()
            review_file.close()
            #
            reviews.append(review)
            sentiment_result.append(label_map[label])
            #
    return (reviews,sentiment_result)
#

print(ds_files_path+"train")
print(ds_files_path+"test")
reviews_train,y_train = load_sentiments_files(ds_files_path+"train")
print(reviews_train[:2])
print(y_train[:2])
reviews_test,y_test = load_sentiments_files(ds_files_path+"test")
print(reviews_test[:2])
print(y_test[:2])
#
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
