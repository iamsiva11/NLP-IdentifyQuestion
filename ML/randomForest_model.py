import pandas as pd
import numpy as np

from string import digits
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def load_data(csv_file):
	names = ["question", "type"]
	dataset_all = pd.read_csv(csv_file,names=names)
	return dataset_all

def preapre_data(dataset_all):
	dataset=dataset_all.values
	X_Train = dataset[:,0]
	Y_Train = dataset[:,1]
	return  X_Train,Y_Train

def remove_questionmark(slist):
    new_x = []
    for x in slist:
         new_x.append(x.replace("?",""))
    return new_x

def remove_numbers(slist):
    res = map(lambda x: x.translate(None, digits), slist)
    return res


def vectorise_featres(X_Train):
    vectorizer = CountVectorizer(analyzer="word", preprocessor=None, tokenizer=None, stop_words=None, max_features=5000)
    train_data_features = vectorizer.fit_transform(X_Train)
    # Numpy arrays are easy to work with, so convert the result to an array
    train_data_features = train_data_features.toarray()
    return vectorizer, train_data_features


def preapare_test_data(X_Test):
	#Preapre, Preprocess Test Data
	X_Test= map(str.lower,X_Test)
	X_Test = remove_questionmark(X_Test)
	X_Test = map(str.rstrip,X_Test)
	X_Test = remove_numbers(X_Test)

	#Test data preapre
	#vectorizer = CountVectorizer(analyzer="word", preprocessor=None, tokenizer=None, stop_words=None, max_features=5000)
	test_data_features = vectorizer.transform(X_Test)
	# Numpy arrays are easy to work with, so convert the result to an array
	test_data_features = test_data_features.toarray()
	#print('The dimension of test_data_features is {}.'.format(test_data_features.shape))    
	return test_data_features


def run_rfClassifier():
    rf_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0) 

    # Use cross validation to evaluate the performance of Random Forest
    rf_clf_error = cross_val_score(rf_clf, train_data_features, encoded_y, 
                                       cv=5, scoring='accuracy', n_jobs=-1).mean()

    print('Random Forest Accuracy: {:.4}'.format(rf_clf_error*100))
    return rf_clf	


def encode_lables(Y_Train):
    encoder = LabelEncoder()
    encoder.fit(Y_Train)
    encoded_y = encoder.transform(Y_Train)
    return encoder, encoded_y

if __name__=="__main__":
	#Load Data
	dataset_all = load_data('nikiai_train.csv')
	
	#Preape Data
	X_Train, Y_Train = preapre_data(dataset_all)

	#Preapre Test data
	X_Test=["Name 11 famous martyrs",
	"Who was the inventor of silly putty ?",
	"What 1920s cowboy star rode Tony the Wonder Horse ?",
	"How many villi are found in the small intestine ?",
	"does this hose have one ?",
	"What is your name?",
	"When is the show happening?",
	"Is there a cab available for airport?",
	"What time does the train leave",
	"when was the last time you did something for the first time" ]	
	test_data_features = preapare_test_data(X_Test)
	
	#Clean Data
	X_Train = remove_questionmark(X_Train)
	X_Train = map(str.rstrip, X_Train)
	X_Train = remove_numbers(X_Train)

	#Encode Labels
	encoder, encoded_y = encode_lables(Y_Train)
	vectorizer, train_data_features = vectorise_featres(X_Train)

	#Fit the Model 
	rf_clf = run_rfClassifier()
	rf_clf.fit(train_data_features, encoded_y)

	#Make Predictions
	y = rf_clf.predict(test_data_features)
	decoded_labels = encoder.inverse_transform(y)
	print decoded_labels 