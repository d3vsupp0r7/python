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
print("*) First review")
print(reviews[0])
print("*) Last review")
print(reviews[-1])

print("*) First 3 review")
print(reviews[:3])
print("*) last 3 review")
print(reviews[-3:])

print("*) First 10 sentiment result")
print(sentiment_result[:10])
print("*) Last 10 sentiment result")
print(sentiment_result[-10:])

##
ds_files_path = "C:/dev/nlp_dataset/nlp_ml_dataset_aclImdb/"

def load_sentiments_files(path,labels=["pos","neg"]):
    reviews = []
    sentiment_result = []
    #labels = ["pos","neg"]
    label_map = {"pos":1,"neg":0}
    param_path = path

    for label in labels:
        #path = ds_files_path+"/"+label
        path += "/"+ label
        print("path search fun: " + path)
        for file in listdir(path):
            review_file = open(path+"/"+file,encoding="utf-8")
            #
            review = review_file.read()
            review_file.close()
            #
            reviews.append(review)
            sentiment_result.append(label_map[label])
        path = param_path
    return (reviews,sentiment_result)
#
print("*) Path for files to analize")
print(ds_files_path+"train")
print(ds_files_path+"test")
print("*) Loading dataset training files")
reviews_train,y_train = load_sentiments_files(ds_files_path+"train")

print("*) First 10 review data")
print(reviews_train[:2])
print(y_train[:2])
print("*) Last 2 review data")
print(reviews_train[-2:])
print(y_train[-2:])
#
print("*) Loading dataset testing files")
reviews_test,y_test = load_sentiments_files(ds_files_path+"test")
print(reviews_test[:2])
print(y_test[:2])
#
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer

'''
CountVectorizer
max_features: define the max number of feature to take into account.
For NLP problem this means the number of common words to take into account.


'''
bag_of_words = CountVectorizer(max_features=5000)
'''
IMPO: sklearn notes

sklearn use two class tipologies:
*) Transformes: transform the data
*) Estimators: class for built our models.
    fit()           : execute calculus
    transform()     : execute the real transformation
    fit_trasform()  : execute both methods
'''
print("## MANAGE TRAIN DATASET")
bag_of_words_train = bag_of_words.fit_transform(reviews_train)
print(type(bag_of_words_train))
X_train = bag_of_words_train.toarray()
print(type(X_train))
print(type(X_train.shape))
#shape: n_rows * n_cols => n_rows(review) * n_cols(features=>5000)
print(X_train.shape)
print("## MANAGE TEST DATASET")
bag_of_words_test = bag_of_words.transform(reviews_test)
X_test =bag_of_words_test.toarray()
print(type(X_test))
print(type(X_test.shape))
print(X_test.shape)
# Data standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#
print("*)Standardization output for train dataset")
'''
Some indicators that a good standardization was executed, are based on values of mean and standard deviation.

*) mean value must be around the 0 value
*) standard deviation must be around the 1 value
'''
print("\t*) MEAN TRAIN DATASET")
print(X_train.mean())
print("\t*) STD DEVIATION TRAIN DATASET")
print(X_train.std())

print("*)Standardization output for test dataset")
print("\t*) MEAN TRAIN DATASET")
print(X_test.mean())
print("\t*) STD DEVIATION TRAIN DATASET")
print(X_test.std())

## MODEL CREATION
print("## MODEL CREATION ##")
'''
Problems:
*) Data to elaborate request more hardware resources and time for calculations!
    ERROR RELATED:
   ...\Miniconda3\lib\site-packages\sklearn\linear_model\_logistic.py:939: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html.
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)

'''
# Logistic regression
print("## NLP with MachineLearning Approach - [sklearn] Logistic Regression")
print("*) NLP Logistic Regression - model training")
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()
'''
Model traing.
We use fit() method.

'''
logistic_regression.fit(X_train,y_train)
print(logistic_regression)

# Evaluate quality of model
print("*) NLP Logistic Regression - evaluate quality of model training")
from sklearn.metrics import log_loss, accuracy_score
'''
log_loss: measures the performance of a classification model where the prediction input is 
a probability value between 0 and 1. The goal of our machine learning models is to minimize 
this value. 

accuracy_score: give as the percentage of correct classification

'''
train_prediction = logistic_regression.predict(X_train)
train_prediction_proba = logistic_regression.predict_proba(X_train)
#accuracy: right_predicted_label, model_predictions
accuracy_score_out = accuracy_score(y_train, train_prediction)
print("*) Accuracy score for TRAIN model")
print(accuracy_score_out)
print("*) LOG_LOSS score for TRAIN model")
#log_loss: right_predicted_label, probability
out_log_loss = log_loss(y_train,train_prediction_proba)
print(out_log_loss)
'''
ANALISYS ON TRAIN DATASET

*) Accuracy score for TRAIN model
0.99448
*) LOG_LOSS score for TRAIN model
0.03557194552320964

'''
print("## ANALYSIS ON TEST DATASET ##")
test_prediction = logistic_regression.predict(X_test)
test_prediction_proba = logistic_regression.predict_proba(X_test)
accuracy_score_test_out = accuracy_score(y_test, test_prediction)
print("*) Accuracy score for TEST model")
print(accuracy_score_test_out)
print("*) LOG_LOSS score for TEST model")
out_log_loss_test = log_loss(y_test,test_prediction_proba)
print(out_log_loss_test)
'''
ANALISYS ON TEST DATASET
*) Accuracy score for TEST model
0.81268
*) LOG_LOSS score for TEST model
2.685705863681372 -> very high: the model is uncertain about the predictions it has made.

How we can see metrics from train and test dataset are very different we have an "OVERFITTING PROBLEM"

'''
print("## MANAGING OVERFITTING PROBLEMS ##")
'''
The overfitting problem is when an machine learning model, instead of learning 
from the data, it tends to store it. if the dataset changes, the predictions on the 
new dataset will be completely wrong.

To manage overfitting we can use some two technics:
*) Reduce model complexity:
    
*) Data regularization:
    serve to add penalty to high coefficients of model that are causing the overfitting.

'''
#############

print("## LINEAR REGRESSION WITH REGULARIZATION ##")
'''

'''
from sklearn.linear_model import LogisticRegression
'''

'''
logistic_regression_reg = LogisticRegression(C=0.001)
logistic_regression_reg.fit(X_train,y_train)
#
train_prediction = logistic_regression_reg.predict(X_train)
train_prediction_proba = logistic_regression_reg.predict_proba(X_train)
#
accuracy_score_reg_out = accuracy_score(y_train, train_prediction)
print("*) Model with regularization")
print("\t*) Accuracy score for TRAIN model [with REGULARIZATION]")
print("\t{}".format(accuracy_score_reg_out) )
print("\t*) LOG_LOSS score for TRAIN model [with REGULARIZATION]")
out_reg_log_loss = log_loss(y_train,train_prediction_proba)
print("\t{}".format(out_reg_log_loss) )
#
print("\t## ANALYSIS ON TEST DATASET [with REGULARIZATION] ##")
test_prediction = logistic_regression_reg.predict(X_test)
test_prediction_proba = logistic_regression_reg.predict_proba(X_test)
accuracy_score_reg_test_out = accuracy_score(y_test, test_prediction)
print("\t*) Accuracy score for TEST model [with REGULARIZATION]")
print("\t{}".format(accuracy_score_reg_test_out) )
print("\t*) LOG_LOSS score for TEST model [with REGULARIZATION]")
out_reg_log_loss_test = log_loss(y_test,test_prediction_proba)
print("\t{}".format(out_reg_log_loss_test) )
'''
OUTPUT whit regularization of 0.001

*) Model with regularization
	*) Accuracy score for TRAIN model
0.94368
	*) LOG_LOSS score for TRAIN model
0.19738890084924213
	## ANALYSIS ON TEST DATASET ##
	*) Accuracy score for TEST model
0.87748
	*) LOG_LOSS score for TEST model
0.3125515298501161
'''
print("\t\tPREVIOUS DATA WITH NO REGULARIZATION")
print("\t\t*************")
print("\t\t*) Accuracy score for TRAIN model")
print("\t\t{}".format(accuracy_score_out) )
print("\t\t*) LOG_LOSS score for TRAIN model")
print("\t\t{}".format(out_log_loss) )

print("\t\t*) Accuracy score for TEST model")
print("\t\t{}".format(accuracy_score_test_out) )
print("\t\t*) LOG_LOSS score for TEST model")
print("\t\t{}".format(out_log_loss_test) )
print("\t\t*************")

############################
############################

## NORMAL VARS
accuracy_score_out
out_log_loss

accuracy_score_test_out
out_log_loss_test

## REGULARIZATION VARS
accuracy_score_reg_out
out_reg_log_loss

accuracy_score_reg_test_out
out_reg_log_loss_test

############################
############################

# TESTING OUR MODEL
print("# Single result prediction for a new review")
good_review_example_01 = "This is the best movie i've ever seen"
#get bag of word model for new review
x = bag_of_words.transform([good_review_example_01])
print("*) Bag of words representation for new review")
print(x)
print("*) Bag of words representation for new review [TYPE]")
print(type(x))
print("*) Apply the model to our new data (our new review)")
#1 => positive review
print(logistic_regression_reg.predict(x))
#Example using numpy representation
bad_review_example_01 = "This is the worst movie i've ever seen"
x_bad_review_01 = bag_of_words.transform([bad_review_example_01])
x_bad_review_01 = x_bad_review_01.toarray()
print(logistic_regression_reg.predict(x_bad_review_01))
#Example using numpy representation
complex_review_example_01 = "This is a good movie. Some choices are very bad and i don't like it. But over all i've appreciate the meanings."
x_mixed_review_01 = bag_of_words.transform([complex_review_example_01])
x_mixed_review_01 = x_mixed_review_01.toarray()
print(logistic_regression_reg.predict(x_mixed_review_01))
################################
print("### PREPROCESSING CORPUS USING NLTK")
import nltk