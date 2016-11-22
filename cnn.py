#! /usr/bin/env python

#coming back to this for the final project

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
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold
from sklearn import preprocessing


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import random
import itertools

FLAGS = None




def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_3x3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='SAME')




def one_hot_encoding (t_ls,num_ls):
    encoding = np.zeros(((t_ls.shape[0]), num_ls))

    for i in range(0, (t_ls.shape[0])):
        li = t_ls[i]
        for j in range(0, num_ls):
            if li == (j + 1):
                encoding[i][j] = 1
    return encoding


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')







train_data = np.loadtxt('Data/caltechTrainData.dat')
test_data = train_data
extra_feats = np.divide(train_data, 255.0)

train_data = np.concatenate((train_data, extra_feats), axis=1)


# normalize the data
train_data = preprocessing.scale(train_data)

train_labels = np.loadtxt('Data/caltechTrainLabel.dat')


num_images_to_plot = 10

for i in range(test_data.shape[0] - (test_data.shape[0] - num_images_to_plot)):

    im_train = test_data[i,:].reshape((30,30,3), order='F')

    plt.imshow(im_train)
    plt.show()


num_examples = train_data.shape[0]
num_features = train_data.shape[1]
num_labels = 18


one_hot_labels = one_hot_encoding(train_labels, num_labels)


# Create the model # deprecated
x = tf.placeholder(tf.float32, [None, num_features])
W = tf.Variable(tf.zeros([num_features, num_labels]))
b = tf.Variable(tf.ones([num_labels]))
y = tf.nn.softmax(tf.matmul(x, W) + b)


# first convulational layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 30, 30, 1])


h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_3x3(h_conv1)

# second convolutional layer

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_3x3(h_conv2)



# densely connected layer

W_fc1 = weight_variable([10*10*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 10*10*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout/regularization

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



# readout layer


W_fc2 = weight_variable([6400, 18])
b_fc2 = bias_variable([18])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)








# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, num_labels])


sess = tf.InteractiveSession()

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
prediction = tf.argmax(y, 1)


training_its = 10

for i in range(training_its):

    num_folds = random.randrange(2,10) # pick a random value for number of folds, k
    kf = KFold(n_splits=num_folds)

    average_performance = 0.0

    for train, test in kf.split(train_data, train_labels):

        batch_xs = train_data[train]
        batch_ys = one_hot_encoding(train_labels[train], num_labels)

        test_xs = train_data[test]
        test_ys = one_hot_encoding(train_labels[test], num_labels)

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob:0.5})
        print(sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys}))

print(prediction.eval(feed_dict={x: train_data}))
print (sess.run(accuracy, feed_dict={x: train_data, y_: one_hot_labels}))