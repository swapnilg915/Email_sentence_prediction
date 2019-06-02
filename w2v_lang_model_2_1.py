# -*- coding: utf-8 -*-
import pickle, traceback, csv, re, os, json
import numpy as np
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
from pymagnitude import *
# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	for _ in range(n_words):
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		yhat = model.predict_classes(encoded, verbose=0)
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		in_text += ' ' + out_word
	return in_text

def loadw2vLocal():
    if not os.path.exists('/home/swapnilg/word2vec/GoogleNews-vectors-negative300.bin.gz'):
        raise ValueError('google word2vec model is not there !! ')
    
    model = KeyedVectors.load_word2vec_format('/home/swapnilg/word2vec/GoogleNews-vectors-negative300.bin.gz', limit=500000, binary=True)
    return model

def loadw2v_magnitude():
    print("\n loading w2v magnitude ... ")
    return Magnitude("../../Downloads/GoogleNews-vectors-negative300.magnitude", pad_to_length = 500000)

#step 6 === create embedding matrix
def createEmbeddingLayer(word_index, w2vmodel, vocab_size):
    embedding_matrix = np.zeros((vocab_size, 300))
    
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        else:
            embedding_vector = np.zeros((1, 300)) # vector of 1 x 300
            try:
                # embedding_vector = w2vmodel[word]
                embedding_vector = w2vmodel.query(word)
            except:
                pass
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    
    print("\n embedding_matrix = ", embedding_matrix.shape)
    embed_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=2)
    return embed_layer


def clean_str(string): # unused
    string = re.sub(r"[^A-Za-z0-9\.\@\?']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

        
def getData():
    csvfile = open("text_classification_dataset.csv", "r")
    dictobj = csv.DictReader(csvfile)
    final_data = ''
    for each in dictobj:
        final_data += each['reviews'] + '\n'
    return final_data

data = getData()
print("\n loaded data === ", len(data))

data = clean_str(data)
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
word_index = tokenizer.word_index
print("\n word_index --- ", word_index)
encoded = tokenizer.texts_to_sequences([data])[0]
# retrieve vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# encode 2 words -> 1 word
sequences = list()
for i in range(2, len(encoded)):
	sequence = encoded[i-2:i+1]
	sequences.append(sequence)
print('\n Total Sequences:', len(sequences))
# pad sequences
max_length = max([len(seq) for seq in sequences])
print('Max Sequence Length:',max_length)
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre', truncating='pre', value=0)
# split into input and output elements
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)

## emb_layer
# w2vmodel = loadw2vLocal()
w2vmodel = loadw2v_magnitude()
embed_layer = createEmbeddingLayer(word_index, w2vmodel, vocab_size)

# define model
model = Sequential()
model.add(embed_layer)
model.add(LSTM(50))
model.add(Dropout(0.3))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, batch_size=128, verbose=2)

#####################################################
#save model
try:  
    model.save('lang_model_2_1.h5')
    print ("\n model saved successfully == ")
    with open('tokenizer_lang_model_2_1.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
        print ("\n tokenizer saved successfully == ")
except Exception as e:
    print ("\n error in model saving = ",e,"\n ",traceback.format_exc())
