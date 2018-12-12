import tensorflow as tf
import numpy as np


class ConvLSTM():
    def __init__(self, num_classes, num_lstm_cells=128, num_lstm_layers=1,
                 kernel_size=(4), filter_size=[128, 256, 128], pool_size=(2),
                 num_cnn_layers=3, dropout_rate=0.1):
        self.num_classes = num_classes
        self.num_lstm_cells = num_lstm_cells
        self.num_lstm_layers = num_cnn_layers
        self.pool_size = pool_size
        self.num_cnn_layers = num_cnn_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        self.model = None

    def create_cnn_model(self, input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv1D(self.filter_size[0], self.kernel_size, input_shape=input_shape))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        for layer in range(1, self.num_cnn_layers):
            model.add(tf.keras.layers.Conv1D(self.filter_size[layer], self.kernel_size))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.AveragePooling1D(self.pool_size))
        model.add(tf.keras.layers.Flatten())
        return model

    def create_lstm_model(self, input_shape):
        model = tf.keras.models.Sequential()
       # model.add(tf.keras.layers.Permute((2,1), input_shape=input_shape))
        model.add(tf.keras.layers.CuDNNLSTM(self.num_lstm_cells,
                  return_sequences=True, input_shape=input_shape))
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        for layer in range(1, self.num_lstm_layers):
            model.add(tf.keras.layers.CuDNNLSTM(self.num_lstm_cells, return_sequences=True))
            model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Flatten())
        return model

    def create_network(self, input_data, time_steps, num_features):
        cnn_input = tf.reshape(input_data, (-1, time_steps, num_features))
        lstm_input = tf.reshape(input_data, (-1, time_steps, num_features))
        shape_cnn = cnn_input.shape[1:]
        shape_lstm = lstm_input.shape[1:]
        lstm_input = tf.keras.layers.Input(shape=shape_lstm, name='lstm_input')
        cnn_input = tf.keras.layers.Input(shape=shape_cnn, name='cnn_input')
        cnn_out = self.create_cnn_model(shape_cnn)(cnn_input)
        lstm_out = self.create_lstm_model(shape_lstm)(lstm_input)
        network_output = tf.keras.layers.concatenate([cnn_out, lstm_out])
        network_output = tf.keras.layers.Dense(self.num_classes,
                                               activation=tf.nn.softmax,
                                               name='network_output'
                                               )(network_output)
        model = tf.keras.models.Model(inputs=[lstm_input, cnn_input],
                                      outputs=[network_output])
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def fit(self, input_data, labels, num_epochs, time_steps, num_features,
            batch_size, learn_rate=0.01):
            cnn_input = np.reshape(input_data, (-1, time_steps, num_features))
            lstm_input = np.reshape(input_data, (-1, time_steps, num_features))
            self.model.fit({'lstm_input': lstm_input, 'cnn_input': cnn_input},
                       {'network_output': labels},
                       epochs=num_epochs, batch_size=batch_size,
                       validation_split=0.2)

    def evaluate(self, test_data, test_labels, time_steps, num_features):
        cnn_data = np.reshape(test_data, (-1, time_steps, num_features))
        lstm_data = np.reshape(test_data, (-1, time_steps, num_features))
        loss, accuracy = self.model.evaluate(x=[lstm_data, cnn_data], y=test_labels, steps=2)
        print("Model loss:", loss, ", Accuracy:", accuracy)
        return loss, accuracy
