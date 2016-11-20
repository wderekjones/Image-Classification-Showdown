#! /usr/bin/env python


# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import train_test_split

import argparse
import numpy as np

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def one_hot_encoding (t_ls,num_ls):
    encoding = np.zeros(((t_ls.shape[0]), num_ls))

    for i in range(0, (t_ls.shape[0])):
        li = t_ls[i]
        for j in range(0, num_ls):
            if li == (j + 1):
                encoding[i][j] = 1
    return encoding



train_data = np.loadtxt('Data/caltechTrainData.dat')
train_labels = np.loadtxt('Data/caltechTrainLabel.dat')
test_data = train_data

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#train_data = mnist.train.images
#train_labels = mnist.train.labels

num_examples = train_data.shape[0]
num_features = train_data.shape[1]
num_labels = 18


one_hot_labels = one_hot_encoding(train_labels,num_labels)


# Create the model
x = tf.placeholder(tf.float32, [None, num_features])
W = tf.Variable(tf.zeros([num_features, num_labels]))
b = tf.Variable(tf.zeros([num_labels]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, num_labels])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.initialize_all_variables().run()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
prediction = tf.argmax(y, 1)

#for i in range(1000):

    # figure a way to split training data and training labels into batches and train over each step
batch_xs, test_xs, batch_ys, test_ys = train_test_split(train_data,train_labels,test_size=0.3,random_state=1)

batch_ys = one_hot_encoding(batch_ys,num_labels)
test_ys = one_hot_encoding(test_ys,num_labels)
sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print(sess.run(accuracy, feed_dict={x: test_xs ,
                                       y_: test_ys }))
print(prediction.eval(feed_dict={x: train_data}))






