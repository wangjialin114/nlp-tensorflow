from __future__ import division
from __future__ import print_function

from utils import imdb
from models import lstm
import tensorflow as tf
import numpy as np
from utils import sequence

class Top(object):
    # model is a string, one of ['lstm', 'cnn']
    def __init__(self, model):
        self.model = model

    def run_lstm_model(self):
        print('loading data')
        config = lstm.Config()
        lstm_model = lstm.ImdbLstm(config)
        (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=config.vocab_size)
        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')
        X_train = sequence.pad_sequences(X_train, maxlen=config.num_steps)
        X_test = sequence.pad_sequences(X_test, maxlen=config.num_steps)

        with tf.Session() as sess:
            lstm_model.run_epoch(sess, X_train, y_train, X_test, y_test)

    #TODO
    def run_cnn_model(self):
         pass

    def run_model(self):
        if self.model == "lstm":
            self.run_lstm_model()
        if self.model == "cnn":
            self.run_cnn_model()

def Test():
    top = Top("lstm")
    top.run_model()

#if __name__ == "main":
Test()