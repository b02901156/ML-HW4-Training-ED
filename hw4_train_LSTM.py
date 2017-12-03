# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 07:49:20 2017

@author: zhewei
"""

from keras.layers.core import Activation, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


DICT_SIZE = 20000
MAX_SENTENCE_LENGTH = 25

EMBEDDING_SIZE = 32
HIDDEN_LAYER_SIZE = 128
BATCH_SIZE = 64
NUM_EPOCHS = 3


def read_train(filename):
    with open(filename,'r', encoding='utf8') as train_data:
        x_train = []
        y_train = []
        for line in train_data:
            label, sentence = line.strip().split(sep = "+++$+++")
            #words = sentence.split()
            y_train.append(label)
            x_train.append(sentence)
    return x_train, y_train

x_train, y_train = read_train("training_label.txt")


# set the total token dictionary size to 5000
token = Tokenizer(DICT_SIZE, filters='\t\n')
token.fit_on_texts(x_train)

# transfer the text to sequences
x_train_seq = token.texts_to_sequences(x_train)

# pad zeros to given length
x_train = sequence.pad_sequences(x_train_seq, maxlen=MAX_SENTENCE_LENGTH)

# Split input into training and test
Xtrain, Xtest, ytrain, ytest = train_test_split(x_train, y_train, test_size=0.5, random_state=42)
'''
# Build model
model = Sequential()
# To embed each word(input_dim) into vector 
model.add(Embedding(input_dim=DICT_SIZE, output_dim=EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
#model.add(Dropout(0.2))
model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.1))
#model.add(LSTM(HIDDEN_LAYER_SIZE))
model.add(Dense(units=128, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
'''
model = Sequential()
model.add(Embedding(input_dim=DICT_SIZE, output_dim=EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(32))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())

history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, 
                    epochs=NUM_EPOCHS,
                    validation_data=(Xtest, ytest))

score,	acc	=	model.evaluate(Xtest,	ytest,	batch_size=BATCH_SIZE)
print("Test	score:	%.3f,	accuracy:	%.3f"	%	(score,	acc))

model.save('model6.h5')
#x_train_seq = token.texts_to_sequences(x_train)



# read test
def read_test(filename):
    with open(filename,'r', encoding='utf8') as test_data:
        x_test = []
        next(test_data)
        for line in test_data:
            index, sentence = line.strip().split(",",1)
            x_test.append(sentence)
    return x_test

x_test = read_test("testing_data.txt")
x_test_seq = token.texts_to_sequences(x_test)
x_test1 = sequence.pad_sequences(x_test_seq, maxlen=30)
x_test3 = sequence.pad_sequences(x_test_seq, maxlen=MAX_SENTENCE_LENGTH)


import csv
def get_result(predict_class, outputs_dir):
    with open(outputs_dir, 'w', newline='') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['id', 'label'])
        index = 0
        for element in predict_class:
            writer.writerow([str(index), str(int(element))])
            index += 1
            
def get_ensemble_result(predict_prob, outputs_dir):
    result = []
    for i in range(len(predict_prob)):
        if predict_prob[i] >= 0.5:
            result.append(int(1))
        else:
            result.append(int(0))
        
    with open(outputs_dir, 'w', newline='') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['id', 'label'])
        index = 0
        for element in result:
            writer.writerow([str(index), str(int(element))])
            index += 1

prediction = model.predict_classes(x_test3)
get_result(prediction, 'results5.csv')

model1 = load_model("model3.h5")
model2 = load_model("model4.h5")
model3 = load_model("model5.h5")

prediction1 = model3.predict(x_test3)
prediction3 = model3.predict(x_test3)

prediction = model.predict(x_test3)
prediction1 = model3.predict(x_test1)
prediction2 = model2.predict(x_test3)

prediction = (prediction + prediction2) / 2
get_ensemble_result(prediction, 'results9.csv')


    