import numpy as np
from ConvLSTM import ConvLSTM
import data_handler as data
import tensorflow as tf
import os


if __name__ == '__main__':
    num_epochs = 5000
    n_classes = 20
    batch_size = 20
    num_features = 19
    timesteps = 150
    rnn_size = 256
    max_len = 150
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    network_input = tf.placeholder(tf.float32, [None, timesteps, num_features])
    network_output = tf.placeholder('float', [None, n_classes])
    inputs, labels = data.open_data(max_len=max_len)
    print(len(inputs), np.array(inputs).shape)
    inputs, labels = np.array(inputs), np.array(labels)
    test_inputs = inputs[int(0.8*len(inputs)):len(inputs)]
    test_labels = labels[int(0.8*len(labels)):len(labels)]
    inputs = inputs[0:int(0.8*len(inputs))]
    labels = labels[0:int(0.8*len(labels))]
    model = ConvLSTM(n_classes)
    inputs = np.array([np.float32(x) for x in inputs])
    print(inputs.dtype)
    model.create_network(network_input, timesteps, num_features)
    model.fit(inputs, labels, test_inputs, test_labels, num_epochs, timesteps, num_features, batch_size)
    model.evaluate(test_inputs, test_labels, timesteps, num_features)
