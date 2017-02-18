import pandas as pd
import numpy as np
from string import digits

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Dropout, Activation
from keras.layers import  Embedding
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import  LSTM, GRU
from keras.layers.convolutional import Convolution1D
from keras.layers import Conv1D, MaxPooling1D

MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 50

#Load Data
def load_data(csv_file):
    names = ["question", "type"]
    dataset_all = pd.read_csv(csv_file,names=names)
    return dataset_all

#Preapre data - get features and labels
def preapre_data(dataset_all):
    dataset=dataset_all.values
    X_Train = dataset[:,0]
    Y_Train = dataset[:,1]
    return  X_Train, Y_Train

# Clean Data
def remove_questionmark(slist):
    new_x = []
    for x in slist:
         new_x.append(x.replace("?",""))
    return new_x
    
def remove_numbers(slist):
    res = map(lambda x: x.translate(None, digits), slist)
    return res

#Encode labels
def encode_labels(Y_Train):
    encoder = LabelEncoder()
    encoder.fit(Y_Train)
    encoded_y = encoder.transform(Y_Train)
    return  encoder, encoded_y

def preapre_testData(X_Test,tok):
    X_Test = map(str.lower,X_Test)
    X_Test = remove_questionmark(X_Test)
    X_Test = map(str.rstrip,X_Test)
    X_Test = remove_numbers(X_Test)
    X_Test = tok.texts_to_sequences(X_Test)
    X_Test_pad = pad_sequences(X_Test, maxlen=MAX_SEQUENCE_LENGTH)
    return  X_Test_pad

def tokenise_padding(X_Train):
    #we construct a tokenizer object, initialized with the number of total terms we want.
    tok = Tokenizer(MAX_NB_WORDS)
    tok.fit_on_texts(X_Train)
    X_Train = tok.texts_to_sequences(X_Train)
    X_Train_pad = pad_sequences(X_Train, maxlen=MAX_SEQUENCE_LENGTH)
    return tok, X_Train_pad

#Model Architecture - 1
def model_vanillaRNN():
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, 32, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Dropout(0.25))
    model.add(SimpleRNN(16, return_sequences=False))
    model.add(Dense(256))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

#Model Architecture - 2
def model_lstm_cnn():
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, 32, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Dropout(0.25))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(LSTM(100))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def evaluate_model(model, train, test):
    scores = model.evaluate(train, test )
    print scores
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

def predict_classes(model, X_Test):
    res = rnn1.predict_classes(X_Test)
    print encoder.inverse_transform(res)    

if __name__ == "__main__":
	dataset_all = load_data('nikiai_train.csv')

	X_Train, Y_Train = preapre_data(dataset_all)

	X_Train = remove_questionmark(X_Train)
	X_Train = map(str.rstrip,X_Train)
	X_Train = remove_numbers(X_Train) 
	tok, X_Train_pad = tokenise_padding(X_Train)

	encoder, encoded_y = encode_labels(Y_Train)
	encoded_y = encoded_y.reshape((-1, 1))

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
	X_Test_pad = preapre_testData(X_Test, tok)	

	#Training
	print "Model 1 - RNN - training starts "
	rnn1 = model_vanillaRNN()
	#print(rnn1.summary())
	rnn1.fit(X_Train_pad, encoded_y, batch_size=32, nb_epoch=20, verbose=1)	
	print "Model 2 - CNN and LSTM - training starts "
	cnn_lstm1 = model_lstm_cnn()
	#print(cnn_lstm1.summary()) 
	cnn_lstm1.fit(X_Train_pad, encoded_y, batch_size=32, nb_epoch=20, verbose=1)
	
	#Splitting dataset for Evaluation
	split_X_train, split_X_test, split_y_train, split_y_test = train_test_split(X_Train_pad, encoded_y, random_state=33) 
	evaluate_model(rnn1, split_X_train, split_y_train)
	evaluate_model(cnn_lstm1, split_X_train, split_y_train)

	# Make Predictions
	#Model 1 Predictions
	predict_classes(rnn1, X_Test_pad)
	#Model 2 Predictions
	predict_classes(cnn_lstm1, X_Test_pad)