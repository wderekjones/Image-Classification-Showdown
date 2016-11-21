from sklearn.model_selection import cross_val_score

from sklearn.ensemble import AdaBoostClassifier

import numpy as np


clf = AdaBoostClassifier(n_estimators=100)

train_data = np.loadtxt('Data/caltechTrainData.dat')
train_labels = np.loadtxt('Data/caltechTrainLabel.dat')
test_data = np.loadtxt('Data/caltechTestData.dat')


scores = cross_val_score(clf,train_data,train_labels)

print (str(scores.mean()))
