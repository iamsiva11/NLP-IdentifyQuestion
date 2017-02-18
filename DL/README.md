#Deep learn Text Model - Keras

# Model - 1 - Vanilla RNN
	
>Architecture
```python
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
```
	
## Model - 2 CNN and LSTM

>Architecture
```python
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, 32, input_length=MAX_SEQUENCE_LENGTH))
model.add(Dropout(0.25))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(100))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

## Input Test Data
	"Name 11 famous martyrs",
	"Who was the inventor of silly putty ?",
	"What 1920s cowboy star rode Tony the Wonder Horse ?",
	"How many villi are found in the small intestine ?",
	"does this hose have one ?",
	"What is your name?",
	"When is the show happening?",
	"Is there a cab available for airport?",
	"What time does the train leave",
	"when was the last time you did something for the first time"

##Accuracy
	Model - 1 - Vanilla RNN accuracy: 99.82%
	Model - 2 - CNN and LSTM accuracy: 99.91%

## Output/Predictions

	Model - 1 - Vanilla RNN
	Predicted Classes: ['unknown' 'who' 'what' 'unknown' 'affirmation' 'what' 'when' 'affirmation'
	 'when' 'unknown']

	Model - 2 CNN and LSTM
	Predicted Classes:  ['unknown' 'who' 'what' 'unknown' 'affirmation' 'what' 'when' 'affirmation'
	 'when' 'unknown']

## Usage 
```python
python IdentifyQuestionType-keras.py
```	 