import numpy as np
from ConvLSTM import ConvLSTM
import data_handler as data
import tensorflow as tf
import os


if __name__ == '__main__':
    num_epochs = 1000
    n_classes = 18
    batch_size = 85
    num_features = 113
    timesteps = 24
    rnn_size = 258
    max_len = 150
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    network_input = tf.placeholder(tf.float32, [None, timesteps, num_features])
    network_output = tf.placeholder('float', [None, n_classes])
    inputs, labels, test_inputs, test_labels = data.load_data()
    print(inputs.shape)
    print(labels.shape)
    model = ConvLSTM(n_classes, num_lstm_cells=rnn_size)
    model.create_network(network_input, timesteps, num_features)
    model.fit(inputs, labels, num_epochs, timesteps, num_features, batch_size)
    model.evaluate(test_inputs, test_labels, timesteps, num_features)
