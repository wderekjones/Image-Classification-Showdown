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
from sklearn.model_selection import KFold
from sklearn import preprocessing

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

train_data = np.loadtxt('Data/caltechTrainData.dat')
test_data = train_data
extra_feats = np.divide(train_data, 255.0)

train_data = np.concatenate((train_data, extra_feats), axis=1)



train_data = preprocessing.scale(train_data)

train_labels = np.loadtxt('Data/caltechTrainLabel.dat')
#test_data = train_data


offset = 10

#for i in range(test_data.shape[0] - (test_data.shape[0] - offset)):

#    im_train = test_data[i,:].reshape((30,30,3), order='F')

#    plt.imshow(im_train)
#    plt.show()




#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#train_data = mnist.train.images
#train_labels = mnist.train.labels

num_examples = train_data.shape[0]
num_features = train_data.shape[1]
num_labels = 18


one_hot_labels = one_hot_encoding(train_labels, num_labels)


# Create the model
x = tf.placeholder(tf.float32, [None, num_features])
W = tf.Variable(tf.zeros([num_features, num_labels]))
b = tf.Variable(tf.zeros([num_labels]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, num_labels])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(.6).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.initialize_all_variables().run()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
prediction = tf.argmax(y, 1)






kf = KFold(n_splits=10)

average_performance = 0.0

for train, test in kf.split(train_data, train_labels):

    batch_xs = train_data[train]
    batch_ys = one_hot_encoding(train_labels[train], num_labels)

    test_xs = train_data[test]
    test_ys = one_hot_encoding(train_labels[test], num_labels)

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print(sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys}))

print(prediction.eval(feed_dict={x: train_data}))
print (sess.run(accuracy, feed_dict={x: train_data, y_: one_hot_labels}))








