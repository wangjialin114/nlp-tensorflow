from __future__ import division
from __future__ import print_function

import sys
from utils import imdb
from models import lstm, cnn, lstm_dynamic_rnn
import tensorflow as tf
import numpy as np
from utils import sequence
from time import time

class Top(object):
    # model is a string, one of ['lstm','lstm_dynamic' 'cnn']
    def __init__(self,model):
        self.model = model
        self.vocab_size = 5000
        self.num_steps = 400
    def load_data(self):
        print('loading data')
        (X_train, self.y_train), (X_test, self.y_test) = imdb.load_data(nb_words=self.vocab_size)
        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')
        self.train_len = map(lambda x: len(x), X_train)
        self.test_len =  map(lambda x: len(x), X_test)
        ## padding the data at the tail
        self.X_train = sequence.pad_sequences(X_train, maxlen=self.num_steps, padding='post')
        self.X_test = sequence.pad_sequences(X_test, maxlen=self.num_steps, padding='post')


    def run_lstm_model(self):
        config = lstm.Config(self.train_len, self.test_len)
        lstm_model = lstm.LSTM(config)

        with tf.Session() as sess:
            lstm_model.run_epoch(sess, self.X_train, self.y_train,self.X_test, self.y_test)

    def run_dynamic_lstm_model(self):
        config = lstm_dynamic_rnn.Config(self.train_len, self.test_len)
        rnn_model = lstm_dynamic_rnn.LSTM_Dynamic(config)

        with tf.Session() as sess:
            rnn_model.run_epoch(sess, self.X_train, self.y_train, self.X_test, self.y_test)

    #
    def run_cnn_model(self):
        config = cnn.Config()
        cnn_model = cnn.CNN(config)
        with tf.Session() as sess:
            cnn_model.run_epoch(sess, self.X_train, self.y_train, self.X_test, self.y_test)

    def run_model(self):
        self.load_data()
        time_start = time()
        if self.model == "lstm":
            self.run_lstm_model()
        if self.model == "dynamic":
            self.run_dynamic_lstm_model()
        if self.model == "cnn":
            self.run_cnn_model()
        if self.model == "rnn_compare":
            self.rnn_compare()
        time_end = time()
        print("run %f seconds" %(time_end-time_start))

    def rnn_compare(self):
        self.load_data()
        time_rnn_start = time()
        with tf.variable_scope("LSTM"):
            self.run_lstm_model()
        time_rnn_end = time()
        with tf.variable_scope("Dynamic_LSTM"):
            time_dynamic_rnn_start = time()
        self.run_dynamic_lstm_model()
        time_dynamic_rnn_end = time()
        print("lstm fixed inputs unrolling : %f seconds, lstm dynamic inputs unrolling: %f seconds"
              %(time_rnn_end-time_rnn_start, time_dynamic_rnn_end-time_dynamic_rnn_start) )


def Test():
    if(len(sys.argv) == 1 ):
        print("usage: python main.py str ")
        print("       str : lstm :for lstm model;")
        print("           : dynamic :for dynamic lstm model")
        print("           : cnn :for cnn model")
        print("           : rnn_compare: compare rnn and dynamic rnn model")
    else:
        model = sys.argv[1]
        top = Top(model)
        top.run_model()


#if __name__ == "main":
Test()
