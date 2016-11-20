import matplotlib.pyplot as plt

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

train_data = np.loadtxt('Data/caltechTrainData.dat')
train_labels = np.loadtxt('Data/caltechTrainLabel.dat')
test_data = np.loadtxt('Data/caltechTestData.dat')

clf = LogisticRegression(solver = 'lbfgs', multi_class='multinomial')


clf.fit(train_data,train_labels)

predictions = clf.predict(train_data)


correct_count = 0.0
for i in range(len(predictions)):
    if (predictions[i] == train_labels[i]):
        correct_count = correct_count + 1


accuracy = float(correct_count) / float(len(predictions))
print (accuracy)
