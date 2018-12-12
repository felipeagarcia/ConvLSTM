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


ACTIVITIES = {
    0: 'no_activity',
    1: 'lying',
    2: 'sitting',
    3: 'standing',
    4: 'walking',
    5: 'running',
    6: 'cycling',
    7: 'nordic_walking',
    9: 'watching_tv',
    10: 'computer_work',
    11: 'car_driving',
    12: 'ascending_stairs',
    13: 'descending_stairs',
    16: 'vaccuum_cleaning',
    17: 'ironing',
    18: 'folding_laundry',
    19: 'house_cleaning',
    20: 'playing_soccer',
    24: 'rope_jumping'
}

def load_pampap2():
    X_train = np.load('../../pamap2/data/X_train.npy')
    X_test = np.load('../../pamap2/data/X_test.npy')
    y_train = np.load('../../pamap2/data/y_train.npy')
    y_test = np.load('../../pamap2/data/y_test.npy')
    return X_train, y_train, X_test, y_test


def shuffle_data_and_labels(data, labels):
    aux = list(zip(data, labels))
    np.random.shuffle(aux)
    data[:], labels[:] = zip(*aux)
    return data, labels
