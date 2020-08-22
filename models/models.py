"""                                 Class for creating deep learning models
Date: 2020-03-19

"""
# loading modules

import pandas as pd
import numpy as np
import sklearn
from keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout, Flatten
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import LSTM, Concatenate, Flatten
from keras.optimizers import Adam
from keras.models import Model
from sklearn.metrics import roc_auc_score
from keras import layers
import re
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_colwidth', -1)
from sklearn.model_selection import train_test_split
# import utils
# from utils import *
# from utils import clean_text
# from utils import pearson_correlation
# from utils import load_embedding, create_embedding_weights, max_seq_len
import keras.backend as K
from keras.optimizers import Adadelta
import tensorflow as tf
from keras.optimizers import Adadelta, Adam
from keras.regularizers import l2

class models():
    """ Contains functions to create deep learning models to classify entailment and also to generate relatedness score
    """
    def __init__(self, embedding_dim, NUM_WORDS, embedding_weights, max_seq_length, task = 'entailment', dropout = None, l2_reg = None):
        """ initializing requrired variables
            Args:
                embedding_dim: embedding dimension
                embedding_weights: matrix with the embedding weights
                max_seq_length: maximum sequence length of input data
                dropout: dropout value between(0 to 1)"""
        self.embedding_dim = embedding_dim
        self.NUM_WORDS = NUM_WORDS
        self.embedding_weights = embedding_weights
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.task = task
        self.l2_reg = l2_reg
        
    def pearson_correlation(self, y_true, y_pred):
        # Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))
        fs_pred = y_pred - K.mean(y_pred)
        fs_true = y_true - K.mean(y_true)
        covariance = K.mean(fs_true * fs_pred)

        stdv_true = K.std(y_true)
        stdv_pred = K.std(y_pred)

        return covariance / (stdv_true * stdv_pred)

    def siames(self):
        """ returns a siames LSTM model"""

        #LSTM layer
        lstm = layers.LSTM(self.embedding_dim)
        
        # Embedding layer with the embedding weights
        embedding_layer = Embedding(
                self.NUM_WORDS,
                self.embedding_dim,
                weights = [self.embedding_weights], 
                input_length = self.max_seq_length,
                trainable = False)
        # Creating inputs
        
        # Left input
        left_input = Input(shape=(self.max_seq_length,), name='input_1')
        left_output = embedding_layer(left_input)
        left_output = lstm(left_output)
        
        # Right input
        right_input = Input(shape=(self.max_seq_length,), name='input_2')
        right_output = embedding_layer(right_input)
        right_output = lstm(right_output) 
        

        # temporary
        dist_normal = lambda x: 1 - K.abs(x[0] - x[1])

        # merging both the input using absolute distance
        # this absolute distance function can be replaced with any other similar measures
        merged = layers.Lambda(function=dist_normal, output_shape=lambda x: x[0], 
                                       name='L1_distance')([left_output, right_output])
        if self.dropout is not None:
            merged = Dropout(self.dropout)(merged)
        
        # output layer
        if self.task.lower() == 'entailment':
            # output for entailment task
            if self.l2_reg is None:
                #without regularization
                predictions = layers.Dense(3, 
                                       activation='sigmoid',
                                       name='entailment_task')(merged)
            else:
                #with regularization
                predictions = layers.Dense(3, 
                                       activation='sigmoid',
                                       name='entailment_task',
                                       kernel_regularizer = l2(self.l2_reg),
                                       bias_regularizer=l2(self.l2_reg)
                                          )(merged)         
        else:
            # output for relatedness task
            if self.l2_reg is None:
                predictions = layers.Dense(1, 
                                       activation='selu',
                                       name= 'similarity_task')(merged)
            else:
                predictions = layers.Dense(1, 
                                       activation='selu',
                                       name= 'similarity_task',
                                        kernel_regularizer = l2(self.l2_reg),
                                       bias_regularizer=l2(self.l2_reg))(merged)
        # Creating the model
        model = Model([left_input, right_input], predictions)
        
        if self.task.lower() == 'entailment':
            #compiling the model for entailment task
            model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
        else:
            #compiling the model for relatedness task
            model.compile(loss='mse', optimizer=Adadelta(), metrics= [self.pearson_correlation])
        print(model.summary())
        return model
    
    def cnn(self, 
            filters=16,
            kernel_size = 3,
            hidden_dims = 250):
        """ Returns a siamese convolutional neural network
            Args:
                 filters: the number of filter for the convolutional layer
                 kernel_size: kernel size
                 hidden_dim: number of hidden dimensions required"""
        # Running using tensorflow.keras
       
        model = Sequential()
        
        # embedding layer for shared weights
        model.add(Embedding(len(self.embedding_weights), self.embedding_dim,weights=[self.embedding_weights], 
                 input_length=self.max_seq_length, 
                 trainable=False ))
        
        # adding dropout
        model.add(Dropout(.1))
        
        #Adding first convolution
        # since we are passing one dimensional sequence, 1D convolution was used
        model.add(Conv1D(filters, kernel_size, padding = 'valid', activation = 'relu'))
        
        # max pooling layer
        model.add(MaxPooling1D())
        
        #2nd convolution layer
        model.add(Conv1D(filters, kernel_size, padding = 'valid', activation = 'relu'))
        
        # max pooling
        model.add(MaxPooling1D())
        
        # flattening the model
        model.add(Flatten())
        
        # adding the relu layer
        model.add(Dense(hidden_dims, activation = 'relu'))
        
        model.add(Dropout(.1))
        
        #left input
        left_input = Input(shape = (self.max_seq_length,))
        left = model(left_input)
        
        #right input
        right_input = Input(shape = (self.max_seq_length,))
        right = model(right_input)

        #merging the models
        l1_layer = layers.Lambda(lambda x: K.abs(x[0] - x[1]))
        l1_distance = l1_layer([left, right])
        
        if self.task.lower() == 'entailment':
            # output for entailment task
            if self.l2_reg is None:
                prediction = Dense(3, 
                                   name = 'entailment_task',
                                   activation= 'sigmoid')(l1_distance)
            else:
                # regularization
                prediction = Dense(3, 
                                   activation= 'sigmoid',
                                   name= 'entailment_task',
                                   kernel_regularizer = l2(self.l2_reg),
                                   bias_regularizer=l2(self.l2_reg))(l1_distance)
        else:
            # output for relatedness task
            if self.l2_reg is None:
                prediction = Dense(1, 
                                   activation= 'selu',
                                  name ='relatedness_task' )(l1_distance)
            else:
                # regularization
                prediction = Dense(1, 
                                   activation= 'selu',
                                   name='relatedness_task',
                                   kernel_regularizer = l2(self.l2_reg),
                                   bias_regularizer=l2(self.l2_reg))(l1_distance)
        
        # creating the model
        siamese_cnn = Model([left_input, right_input], prediction)
        if self.task.lower() == 'entailment':
            #compiling the model for entailment task
            siamese_cnn.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        else:
            #compiling the model for relatedness task
            siamese_cnn.compile(loss='mse', optimizer =Adam(), metrics = [self.pearson_correlation])
            
        print(siamese_cnn.summary())
        
        return siamese_cnn
    
    def GRU_tf(self):
        
        """ Returns bidirection GRU model"""
        
        #creating the model
        model = tf.keras.Sequential([
        
            # embedding layer
        tf.keras.layers.Embedding(len(self.embedding_weights), self.embedding_dim, weights = [self.embedding_weights]),
         
            # BiDirection GRU
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.embedding_dim)),
        
        tf.keras.layers.Dense(self.embedding_dim, activation='relu'),
        
        # prediction layer
        tf.keras.layers.Dense(3, activation='sigmoid')
                                    ])
        print(model.summary())
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return model
    
    def lstm_bi(self):
        main_input = Input(shape=(self.max_seq_length,), dtype='int32', name='main_input') #(N,70)
        #x = Embedding(output_dim=opts['emb'], input_dim=len(VOCABULARY.keys())+1, input_length=N, name='x')(main_input)

        x = Embedding(
                    self.NUM_WORDS,
                    self.embedding_dim,
                    weights = [self.embedding_weights], 
                    input_length = self.max_seq_length,
                    trainable = False)(main_input)

        drop_out = Dropout(0.3, name='dropout')(x) # 70,50
        lstm_fwd = LSTM(125, return_sequences=True, name='lstm_fwd')(drop_out)
        lstm_bwd = LSTM(125, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out)
        #70,100
        bilstm = Concatenate()([lstm_fwd,lstm_bwd])
        #70,200
        drop_out = Dropout(0.1, name="d_bilstm")(bilstm)
        flat_h_star = Flatten(name="flat_h_star")(drop_out)
        out = Dense(3, activation='sigmoid')(flat_h_star)

        model = Model([main_input], output=out)
        model.summary()
        model.compile(loss='binary_crossentropy',optimizer=Adam(), metrics = ['acc'])
        return model
