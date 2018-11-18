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


def load_pampap2():
    X_train = np.load('../../pamap2/X_train.npy')
    X_test = np.load('../../pamap2/X_test.npy')
    y_train = np.load('../../pamap2/y_train.npy')
    y_test = np.load('../../pamap2/y_test.npy')
    return X_train, y_train, X_test, y_test


def shuffle_data_and_labels(data, labels):
    aux = list(zip(data, labels))
    np.random.shuffle(aux)
    data[:], labels[:] = zip(*aux)
    return data, labels
