# -*- coding: utf-8 -*-
'''
Created on Aug 01 2018

@author : Felipe Aparecido Garcia
@github: https://github.com/felipeagarcia/

Open and prepare the data
'''

import numpy as np
import csv
from sklearn.preprocessing import normalize as norm
import pickle


def load_data():
    with open("../../opportunity/xtrain.dat", 'rb') as f:
        X_train = pickle.load(f)
    with open("../../opportunity/xtest.dat", 'rb') as f:
        X_test = pickle.load(f)
    with open("../../opportunity/ytrain.dat", 'rb') as f:
        y_train = pickle.load(f)
    with open("../../opportunity/ytest.dat", 'rb') as f:
        y_test = pickle.load(f)
    X_train, y_train = shuffle_data_and_labels(X_train, y_train)
    X_test, y_test = shuffle_data_and_labels(X_test, y_test)
    return X_train, y_train, X_test, y_test


def shuffle_data_and_labels(data, labels):
    aux = list(zip(data, labels))
    np.random.shuffle(aux)
    data[:], labels[:] = zip(*aux)
    return data, labels
