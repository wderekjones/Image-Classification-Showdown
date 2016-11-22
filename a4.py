# This file has been deprecated

import matplotlib.pyplot as plt
import numpy as np

import itertools

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix





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
train_labels = np.loadtxt('Data/caltechTrainLabel.dat')
test_data = np.loadtxt('Data/caltechTestData.dat')

num_labels = 18

clf = LogisticRegression(solver = 'lbfgs', multi_class='multinomial')


clf.fit(train_data,train_labels)

predictions = clf.predict(train_data)


correct_count = 0.0
for i in range(len(predictions)):
    if (predictions[i] == train_labels[i]):
        correct_count = correct_count + 1


accuracy = float(correct_count) / float(len(predictions))
print (accuracy)


confusion = confusion_matrix(train_labels,predictions)

class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']

plt.figure()

plot_confusion_matrix(confusion,class_names)

plt.show()




