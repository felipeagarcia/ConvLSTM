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


def open_data(max_len, num_activities=20, num_seqs=10):
    data = []
    labels = []
    for i in range(num_activities):
        for j in range(num_seqs):
            i += 1
            i = str(i)
            if len(i) == 1:
                i = '0' + i
            j += 1
            j = str(j)
            if len(j) == 1:
                j = '0' + j
            with open('../data/act' + i + 'seq' + j + '.csv', 'r') as file:
                aux = list(csv.reader(file))
                aux = [np.array(list(map(np.float32, x)), np.float32) for x in aux]
                while(len(aux) < max_len):
                    aux.insert(-1, np.zeros(19))
                aux = aux[:max_len]
                data.append(aux)
                label = np.zeros(num_activities)
                label[int(i) - 1] = 1
                labels.append(label)
            i = int(i)
            j = int(j)
            i -= 1
            j -= 1
    data, labels = shuffle_data_and_labels(data, labels)
    data = [norm(x) for x in data]
    return data, labels


def shuffle_data_and_labels(data, labels):
    aux = list(zip(data, labels))
    np.random.shuffle(aux)
    data[:], labels[:] = zip(*aux)
    return data, labels


def prepare_data(data, labels, length):
    '''
    This function makes every line of the data to be of the same length
    @param data: raw_data to be prepared
    @param labels: labels of the data
    @param length: desired time step
    '''
    prepared_data = []
    prepared_data.append([])
    current_label = labels[0][0]
    pos = 0
    prepared_labels = []
    aux = np.zeros(6)
    aux[current_label - 1] = 1
    prepared_labels.append(list(aux))
    for i in range(len(data)):
        if(current_label == labels[i][0]):
            prepared_data[pos].append(data[i])
        else:
            current_label = labels[i][0]
            # fill the first positions of the data with zeros
            while(len(prepared_data[pos]) < length):
                prepared_data[pos].insert(0, list(np.zeros(561)))
            prepared_data.append([])
            aux = np.zeros(6)
            aux[current_label - 1] = 1
            prepared_labels.append(list(aux))
            pos += 1
    while(len(prepared_data[pos]) < length):
        prepared_data[pos].insert(0, list(np.zeros(561)))
    aux = list(zip(prepared_data, prepared_labels))
    np.random.shuffle(aux)
    prepared_data[:], prepared_labels[:] = zip(*aux)
    return prepared_data, prepared_labels
