# Copyright 2016 WangJialin. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

'''
this model is used to evaluated the imdb reviews sentiment.

'''
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from utils import imdb
from utils import sequence

#when debug set max_epochs = 1


class ImdbLstm(object):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embed_size = 50
        self.add_placeholders()
        self.max_epochs = 3
        self.initial_state_placeholder = tf.placeholder(tf.float32)
        inputs = self.add_embed_layer()

        ##initial state
        cell = self.add_rnn_model()
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        #self.sequence_length_placeholder = tf.placeholder(tf.int32)

        # state is the final state
        outputs, state = tf.nn.rnn(cell, inputs, initial_state=self.initial_state)
        self.final_state = state[-1]
        # add projection layers
        W = tf.get_variable('Weights', shape=[self.hidden_size, 1])
        b = tf.get_variable('Bias', shape=[1])

        y_pred = tf.squeeze(tf.matmul(outputs[-1], W)) + b

        y_pred_sigmoid = tf.sigmoid(y_pred)
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(y_pred, self.label_placeholder)

        self.train_op = self.add_train_op()

        self.correct_num = self.evaluation(y_pred_sigmoid)


        self.summary_op = tf.merge_all_summaries()



    # build the graph functions
    # add the placeholders
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_steps])
        self.label_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size])

    # create the feed_dict
    def create_feed_dict(self, input_batch, label_batch):
        feed_dict = {self.input_placeholder: input_batch,
                     self.label_placeholder: label_batch}
        return feed_dict

    def add_embed_layer(self):
        with tf.device('/cpu:0'), tf.variable_scope('embed'):
            embed = tf.get_variable(name="Embedding", shape=[self.vocab_size, self.embed_size])
            inputs = tf.nn.embedding_lookup(embed, self.input_placeholder)
            inputs = [tf.squeeze(input, squeeze_dims=[1]) for input in tf.split(1, self.num_steps, inputs)]
            #inputs = tf.transpose(inputs, perm=[0,2,1])
        return inputs

    ## add training op
    def add_train_op(self):
        train_op = tf.train.AdamOptimizer(0.0002).minimize(self.loss)
        return train_op

    ## add rnn model
    def add_rnn_model(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=0.0)

        return lstm_cell

    # evalate the prediction
    def evaluation(self, y_pred_sigmoid):
        y_pred_label = (y_pred_sigmoid > 0.5)
        label_placeholder_bool = tf.cast(self.label_placeholder, tf.bool)
        correct_pred_num = []
        correct_pred_num.append(tf.reduce_sum(tf.cast(tf.equal(y_pred_label, label_placeholder_bool), tf.int32)))
        correct_pred_num = np.sum(correct_pred_num)
        return correct_pred_num

    def do_evaluation(self, sess, X, y):
        total_correct_num = 0
        num_steps = len(X) // self.batch_size
        init_state = sess.run([self.initial_state])
        for step in range(num_steps):
            # generate the data feed dict
            input_batch = X[step * self.batch_size:(step + 1) * self.batch_size, :]
            label_batch = y[step * self.batch_size:(step + 1) * self.batch_size]


            feed = {self.input_placeholder: input_batch, self.label_placeholder: label_batch,
                    self.initial_state_placeholder: init_state}
            init_state, correct_num_step = sess.run([self.final_state, self.correct_num], feed)
            total_correct_num += correct_num_step
        print('Testing Accuracy: %f' % (total_correct_num / (num_steps * self.batch_size)))


    def run_epoch(self, sess, X_train, y_train, X_test, y_test):

        # state_step = initial_state

        summary_writer = tf.train.SummaryWriter("data/", sess.graph)
        sess.run(tf.initialize_all_variables())
        for epoch in range(self.max_epochs):
            print('%d Epoch starts, Training....' % (epoch))
            mean_loss = []
            total_correct_num = 0
            state = sess.run([self.initial_state])
            for step in range(len(X_train) // self.batch_size):
                # generate the data feed dict
                input_batch = X_train[step * self.batch_size:(step + 1) * self.batch_size, :]
                label_batch = y_train[step * self.batch_size:(step + 1) * self.batch_size]

                feed = {self.input_placeholder: input_batch, self.label_placeholder: label_batch,
                        self.initial_state_placeholder: state}
                _, state, correct_num_step, loss_step = sess.run(
                    [self.train_op, self.final_state, self.correct_num, self.loss], feed)

                loss_step = np.sum(loss_step)
                mean_loss.append(loss_step)
                total_correct_num += correct_num_step

                if step % 100 == 0:
                    print('step %d / %d : loss : %f' % (step, len(X_train) // self.batch_size, np.mean(mean_loss)))
                    mean_loss = []
            print('precision: %f' % (total_correct_num / len(X_train)))
            print('Testing....')
            self.do_evaluation(sess, X_test, y_test)



class Config(object):
    batch_size = 40
    vocab_size = 5000
    hidden_size = 100
    num_steps = 400

