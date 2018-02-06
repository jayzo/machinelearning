
# coding: utf-8
from sklearn import datasets
import numpy as np
home = "/Users/zhili/src/zuojie/machinelearning/kaggle/data-science-london-scikit-learn"

test = np.loadtxt(home + "/data/test/test.csv", delimiter=',')
label = np.loadtxt(home + "/data/train/trainLabels.csv", delimiter=',')
train = np.loadtxt(home + "/data/train/train.csv", delimiter=',')

print(test[0:2])
print(label[0:2])
print(train[0:2])

