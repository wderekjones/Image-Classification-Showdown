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

import argparse
import numpy as np

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None



train_data = np.loadtxt('Data/caltechTrainData.dat')
train_labels = np.loadtxt('Data/caltechTrainLabel.dat')
test_data = train_data



num_examples = train_data.shape[0]

print (num_examples)

num_features = train_data.shape[1]

print (num_features)

print (train_labels.shape[0])



# Create the model
x = tf.placeholder(tf.float32, [None, num_features])
W = tf.Variable(tf.zeros([num_features, 18]))
b = tf.Variable(tf.zeros([18]))

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [num_features,1])


y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)


sess.run(train_step, feed_dict={x: train_data, y_: train_labels})


#for i in range(10):
#    batch_xs, batch_ys = tf.train()


  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
#  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
#  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#  sess = tf.InteractiveSession()
  # Train
#  tf.initialize_all_variables().run()
#  for _ in range(1000):
#    batch_xs, batch_ys = mnist.train.next_batch(100)
#    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
#  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
#                                      y_: mnist.test.labels}))

#if __name__ == '__main__':
#    main()
  #parser = argparse.ArgumentParser()
  #parser.add_argument('--data_dir', type=str, default='/tmp/data',
  #                    help='Directory for storing data')
  #FLAGS = parser.parse_args()
  #tf.app.run()
