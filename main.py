""" The main python file that calls all the required modules and outputs result.txt with the entailment and relatedness predictions

DATE: 20-03-2020

Group:
Harikrishnan Narayanan
Dhyanesh
"""

#import statements
import pandas as pd
import numpy as np
import keras
import nltk
from nltk.tokenize import word_tokenize
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing required functions from keras
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from sklearn.metrics import roc_auc_score
from keras import layers
import keras.backend as K

#importing libraries required for preprocessing
import re
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_colwidth', -1)
from sklearn.model_selection import train_test_split

# importing the required util function
import functions.utils
from functions.utils import *
from functions.utils import clean_text
from functions.utils import load_embedding, create_embedding_weights, max_seq_len
from keras.utils import to_categorical, plot_model

import functions.model_evaluation
from functions.model_evaluation import *

import functions.data_manipulation
from functions.data_manipulation import *


# loading datasets
#df = pd.read_csv('../sick.csv')
train = pd.read_csv('data/train.txt', delimiter = '\t')
test = pd.read_csv('data/test.txt', delimiter = '\t')
trial = pd.read_csv('data/trial.txt', delimiter = '\t')
test_ann = pd.read_csv('data/test_anno.txt', delimiter = '\t')

######DATA PREPROCESSING#######

#Cleaning the data
train['sentence_A'] = train['sentence_A'].apply(clean_text)
train['sentence_B'] = train['sentence_B'].apply(clean_text)
test['sentence_A'] = test['sentence_A'].apply(clean_text)
test['sentence_B'] = test['sentence_B'].apply(clean_text)
trial['sentence_A'] = trial['sentence_A'].apply(clean_text)
trial['sentence_B'] = trial['sentence_B'].apply(clean_text)

#encoding the target feature
lbl_enc = LabelEncoder()
train['entailment_encoded'] = lbl_enc.fit_transform(train['entailment_judgment'])
trial['entailment_encoded'] = lbl_enc.fit_transform(trial['entailment_judgment'])
############################For now############################################
test_ann['entailment_encoded'] = lbl_enc.fit_transform(test_ann['entailment_judgment'])

#loading the embedding
file_name = 'word_embeddings/glove.6B.300d.txt'
embeddings = load_embedding(file_name)


#tokenizing 
NUM_WORDS = len(embeddings) #200000
sentences = (list(train['sentence_A']) + list(train['sentence_B']) + 
                       list(test['sentence_A']) + list(test['sentence_B'])+ 
                           list(trial['sentence_A']) + list(trial['sentence_B']))
tokenize = Tokenizer(num_words = NUM_WORDS)
tokenize.fit_on_texts(sentences)
sent1_word_seq = tokenize.texts_to_sequences(train['sentence_A'])
sent2_word_seq = tokenize.texts_to_sequences(train['sentence_B'])
sent1_word_seq_test = tokenize.texts_to_sequences(test['sentence_A'])
sent2_word_seq_test = tokenize.texts_to_sequences(test['sentence_B'])
sent1_word_seq_trial = tokenize.texts_to_sequences(trial['sentence_A'])
sent2_word_seq_trial = tokenize.texts_to_sequences(trial['sentence_B'])
word_index = tokenize.word_index

#Matrix with the embedding weights
embedding_dim = 300
embedding_weights = create_embedding_weights(embeddings, embedding_dim, word_index, NUM_WORDS)

# extracting the maximum sequence length
max_seq_length = max_seq_len(sent1_word_seq)
max_seq_length = max_seq_len(sent2_word_seq, max_seq_length)
max_seq_length = max_seq_len(sent1_word_seq_test, max_seq_length)
max_seq_length = max_seq_len(sent2_word_seq_test, max_seq_length)
max_seq_length = max_seq_len(sent1_word_seq_trial, max_seq_length)
max_seq_length = max_seq_len(sent2_word_seq_trial, max_seq_length)

# padding the sequences
sent1_data = pad_sequences(sent1_word_seq, maxlen = max_seq_length)
sent2_data = pad_sequences(sent2_word_seq, maxlen = max_seq_length)

sent1_data_trial = pad_sequences(sent1_word_seq_trial, maxlen = max_seq_length)
sent2_data_trial = pad_sequences(sent2_word_seq_trial, maxlen = max_seq_length)

sent1_data_test = pad_sequences(sent1_word_seq_test, maxlen = max_seq_length)
sent2_data_test = pad_sequences(sent2_word_seq_test, maxlen = max_seq_length)
NUM_WORDS = len(embedding_weights)

labels = to_categorical(np.asarray(train['entailment_encoded']))
labels_valid = to_categorical(np.asarray(trial['entailment_encoded']))


#######################################PART A#########################################################################
### entailment

# number of iterations, batch_size and validation split for the model
batch_size = 10
epochs = 10
# if validation data is not availble, this can be uncommented and passed as parameter while training
#validation_split = .1 


#laoding the module from deep learning models
import models.models as models

# the dropout and l2_reg can be uncommented, to test it with drop out and l2 regularization
m1 = models.models(embedding_dim,
                   NUM_WORDS, 
                   embedding_weights,
#                   dropout = .1,
#                   l2_reg = .0001
                   max_seq_length,
                  )
# building and compiling the model
model = m1.siames()

#training the model
hist = model.fit([sent1_data, sent2_data], labels, batch_size = batch_size, 
                 epochs = epochs,
                 validation_data= ([sent1_data_trial, sent2_data_trial],labels_valid ))

#testing the model
k = model.predict([sent1_data_test, sent2_data_test])
result = convert_prob(k)

prec, rec, f1_s, acc = evaluate_model(test_ann['entailment_encoded'], result)
print("Precision: "+ str(prec))
print("Recall: "+ str(rec))
print("F1 Score: "+ str(f1_s))
print("Accuracy: "+ str(acc))

# dataframe with pair_ID and predicted entailment_judgment
df_entailment = map_entailment(lbl_enc, result, test)


####################################PART B#############################################
# relatedness task


# loading the model
m1 = models.models(embedding_dim = embedding_dim,
                   NUM_WORDS = NUM_WORDS,
                   embedding_weights = embedding_weights,
                   max_seq_length = max_seq_length,
                   task = 'relatedness',
                   dropout = .1,
                   l2_reg=.0001
                  )

model = m1.siames()

hist = model.fit([sent1_data, sent2_data], train['relatedness_score'], batch_size = 10, 
                 epochs = 25,
                 validation_data = ([sent1_data_trial, sent2_data_trial], trial['relatedness_score']))

k = model.predict([sent1_data_test, sent2_data_test])

y_pred = [i[0] for i in k]

y_true = test_ann['relatedness_score']

pearson, spearman, mean_abs_deviation = evaluate_relatedness(y_true, y_pred)
print("pearson corr: "+str(pearson))
print("spearman corr: "+str(spearman[0]))
print("mean_absolute_deviation: " +str(mean_abs_deviation))
# dataframe with pair_ID and predicted entailment_judgment and predicted relatedness_score
df_output = map_relatedness(df_entailment, y_pred)

# writing the output to the output directory
output_csv(df_output, "output/result.csv", index = False)

print("**************************************************************")
print("Execution Completed: The result.csv is saved to output folder")
print("**************************************************************")