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


# Adapted Tensorflow MNIST example (pieces from introduction and deep learning for experts)



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction import image

from sklearn.model_selection import KFold
from sklearn import preprocessing


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import random
import itertools


def one_hot_encoding (t_ls, num_ls):
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
test_data = np.loadtxt('Data/caltechTestData.dat')


train_data = np.divide(train_data, 255.0)

test_data = np.divide(test_data, 255.0)


# output the test images
#num_images_to_plot = 20

#for i in range(test_data.shape[0] - (test_data.shape[0] - num_images_to_plot)):

#    im_train = test_data[i,:].reshape((30,30,3), order='F')

#    plt.imshow(im_train)
#    plt.show()



extra_feats1 = train_data/np.mean(train_data)
extra_feats2 = test_data/np.mean(test_data)


train_data = np.concatenate((train_data, extra_feats1), axis=1)
test_data = np.concatenate((test_data,extra_feats2),axis = 1)

# normalize the data
train_data = preprocessing.scale(train_data)

train_labels = np.loadtxt('Data/caltechTrainLabel.dat')





num_examples = train_data.shape[0]
num_features = train_data.shape[1]
num_labels = 18


one_hot_labels = one_hot_encoding(train_labels, num_labels)


# Create the model
x = tf.placeholder(tf.float32, [None, num_features])
W = tf.Variable(tf.zeros([num_features, num_labels]))
b = tf.Variable(tf.ones([num_labels]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, num_labels])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
#train_step = tf.train.GradientDescentOptimizer(.6).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.9).minimize(cross_entropy)


# dropout to prevent over-fitting
keep_prob = tf.placeholder("float")
y_drop = tf.nn.dropout(y, keep_prob)


sess = tf.InteractiveSession()

tf.initialize_all_variables().run()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, (tf.float32)))

prediction = tf.argmax(y, 1)

keep = 0.9

training_its = 5000

for i in range(training_its):

    num_folds = random.randrange(2,10) # pick a random value for number of folds, k
    kf = KFold(n_splits=num_folds,shuffle=True)

    average_performance = 0.0

    for train, test in kf.split(train_data, train_labels):

        batch_xs = train_data[train]
        batch_ys = one_hot_encoding(train_labels[train], num_labels)

        test_xs = train_data[test]
        test_ys = one_hot_encoding(train_labels[test], num_labels)

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: keep})
        print(sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys}))


print ('----------------------------------------------------------------------')

print ('accuracy: ')
print (sess.run(accuracy, feed_dict={x: train_data, y_: one_hot_labels}))

predictions = prediction.eval(feed_dict={x: train_data})

# due to one-hot-encoding, each prediction is 'off' by 1 less than true value, while this is controlled during the actual training/testing phase, causes problems if not fixed for confusion matrix

for i in range(len(predictions)):
    predictions[i] += 1


confusion = confusion_matrix(train_labels, predictions)


class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']


plt.figure()

plot_confusion_matrix(confusion, class_names, normalize=True)

plt.savefig('confusion_matrix.jpg',dpi = 300)


plt.show()


test_set_predictions = prediction.eval(feed_dict={x: test_data})

results = open('derek_jones_hw4_predictions.txt', mode='w')

for i in range(len(test_set_predictions)):
    test_set_predictions[i] += 1
    results.write(str(test_set_predictions[i]))
    results.write('\n')

results.close()




