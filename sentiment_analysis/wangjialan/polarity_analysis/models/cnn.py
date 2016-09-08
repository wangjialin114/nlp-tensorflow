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


class CNN(object):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embed_size = 50
        self.add_placeholders()
        # when debug set max_epochs = 1
        self.max_epochs = 1
        # Embedding Layer
        self.add_placeholders()
        # Embed layer
        inputs = self.add_embed_layer()
        # CNN layer
        W_cnn = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 50, 64], stddev=0.1))
        b_cnn = tf.Variable(initial_value=tf.truncated_normal(shape=[64], stddev=0.1))
        cnn1_out = tf.nn.relu(self.conv1d(inputs, W_cnn) + b_cnn)
        # Pooling layer
        # pool_out: [batch_size, height/2, channels
        pool_out = self.max_pool1d(cnn1_out)
        pool_out_flat = tf.reshape(pool_out, [-1, 400 * 64])
        ##Fully connected layer
        W_fc = tf.Variable(initial_value=tf.truncated_normal(shape=[400 * 64, 100], stddev=0.1))
        b_fc = tf.Variable(initial_value=tf.truncated_normal(shape=[100], stddev=0.1))
        fc_out = tf.nn.relu(tf.matmul(pool_out_flat, W_fc) + b_fc)
        ##sigmoid layer
        W_classifier = tf.Variable(initial_value=tf.truncated_normal(shape=[100, 1], stddev=0.1))
        b_classifier = tf.Variable(initial_value=tf.truncated_normal(shape=[1], stddev=0.1))
        ##
        y_pred = tf.squeeze(tf.matmul(fc_out, W_classifier) + b_classifier)
        self.y_pred_sigmoid = tf.nn.sigmoid(y_pred)
        ##
        self.evaluation()

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(y_pred, self.label_placeholder)

        self.add_train_op()
    
    
    # build the graph functions
    # add the placeholders
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_steps])
        self.label_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size])

    
    # create the feed_dict
    def create_feed_dict(self, input_batch, label_batch):
        feed_dict = {self.input_placeholder: input_batch,
                    self.label_placeholder:label_batch}
        return feed_dict
    
    def add_embed_layer(self):
        with tf.device('/cpu:0'):
            embed = tf.get_variable(name="Embedding", shape=[self.vocab_size, self.embed_size])
            inputs = tf.nn.embedding_lookup(embed, self.input_placeholder)
            return inputs
        
    ## add training op
    def add_train_op(self):
        self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)
    
    # evalate the prediction 
    def evaluation(self):
        y_pred_label = (self.y_pred_sigmoid > 0.5)
        label_placeholder = tf.cast(self.label_placeholder, tf.bool)
        correct_pred_num = []
        correct_pred_num.append(tf.reduce_sum(tf.cast(tf.equal(y_pred_label, label_placeholder), tf.int32)))
        self.correct_num =  np.sum(correct_pred_num)
    
    def do_evaluation(self, sess, X, y):
        total_correct_num = 0
        num_steps = len(X) // self.batch_size
        for step in range(num_steps):
            # generate the data feed dict
            input_batch = X[step*self.batch_size:(step+1)*self.batch_size, :]
            label_batch = y[step*self.batch_size:(step+1)*self.batch_size]

            feed = {self.input_placeholder:input_batch, self.label_placeholder:label_batch }
            correct_num_step = sess.run([self.correct_num], feed)
            total_correct_num += correct_num_step[0]
        print('Testing Accuracy: %f' %(total_correct_num/(num_steps*self.batch_size)))
        
        
    def conv1d(self, x, W):
        return tf.nn.conv1d(x, W, stride=1, padding="SAME")
    
    ## the tf lib not contain the max_pool1d,
    ## we can add one dummy dimension in the input tensor
    ## x should be [batch_size, height, width, channels]
    ## x : [batch_size, height, channels]
    ## => [batch_size, height, channels, 1]
    ## => [batch_size, height, 1, channels]
    def max_pool1d(self, x):
        x = tf.expand_dims(x, -1)
        x = tf.transpose(x,perm=[0,1,3,2]) ##
        ## invoke maxpool
        pool_out = tf.nn.max_pool(x, ksize=[1,2,1,1], strides=[1,4,1,1], padding="SAME")
        #pool_out = tf.transpose(pool_out,perm=[0,2,1,3])
        pool_out = tf.squeeze(x, squeeze_dims=[2])
        return pool_out
    
    
    def run_epoch(self, sess, X_train, y_train, X_test, y_test):
        #state_step = initial_state
        sess.run(tf.initialize_all_variables())
        for epoch in range(self.max_epochs):
            print('%d Epoch starts, Training....' %(epoch))
            mean_loss = []
            total_correct_num = 0
            for step in range(len(X_train) // self.batch_size):

                input_batch = X_train[step*self.batch_size:(step+1)*self.batch_size, :]
                label_batch = y_train[step*self.batch_size:(step+1)*self.batch_size]

                feed = self.create_feed_dict(input_batch, label_batch)
                '''
                inputs_step, cnn1_out_step, pool_out_step = sess.run([inputs, cnn1_out, pool_out], feed)
                print(inputs_step.shape)
                print(cnn1_out_step.shape)
                print(pool_out_step.shape)
                '''
                _, loss_step, correct_num_step= sess.run([self.train_op, self.loss, self.correct_num], feed)

                loss_step = np.sum(loss_step)
                mean_loss.append(loss_step)
                total_correct_num += correct_num_step

                if step % 100 == 0:
                    print('step %d / %d : loss : %f' %(step, len(X_train) // self.batch_size, np.mean(mean_loss)))
                    mean_loss = []

            print('precision: %f' %(total_correct_num/len(X_train)))
            print('Testing....')
            self.do_evaluation(sess, X_test, y_test)


class Config(object):
    batch_size = 20
    vocab_size = 5000
    hidden_size = 100
    num_steps = 400