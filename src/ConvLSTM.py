import tensorflow as tf
import numpy


class ConvLSTM():
    def __init__(self, num_classes, num_lstm_cells=128, num_lstm_layers=1,
                 cnn_filters=3, pool_size=3, num_cnn_cells=128,
                 num_cnn_layers=2, dropout_rate=0.2):
        self.num_lstm_cells = num_lstm_cells
        self.num_lstm_layers = num_cnn_layers
        self.cnn_filters = cnn_filters
        self.pool_size = pool_size
        self.num_cnn_layers = num_cnn_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None

    def create_cnn_model(self, input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv1D(self.num_cnn_cells, self.cnn_filters,
                  input_shape=input_shape))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling1D(self.pool_size))
        for layer in range(1, self.num_cnn_layers):
            model.add(tf.keras.layers.Conv1D(self.num_cnn_cells,
                      self.cnn_filters))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.MaxPooling1D(self.pool_size))
        model.add(tf.keras.layers.Flatten())
        return model

    def create_lstm_model(self, input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.CuDNNLSTM(self.num_lstm_cells,
                  input_shape=input_shape, return_sequences=True))
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        for layer in range(1, self.num_lstm_layers):
            model.add(tf.keras.layers.CuDNNLSTM(self.num_lstm_cells))
            model.add(tf.keras.layers.Dropout(self.dropout_rate))
        return model

    def create_network(self, input_data, time_steps, num_features):
        input_data = np.array(input_data)
        cnn_input = np.reshape(input_data, (-1, time_steps, num_features, 1))
        lstm_input = np.reshape(input_data, (-1, time_steps, num_features))
        shape_cnn = cnn_input.shape[1:]
        shape_lstm = lstm_input.shape[1:]
        cnn_out = self.create_cnn_model(shape_cnn)(cnn_input)
        lstm_out = self.create_lstm_model(shape_lstm)(lstm_input)
        network_output = tf.keras.layers.concatenate([cnn_out, lstm_out])
        network_output = tf.keras.layers.Dense(self.num_classes,
                                               activation=tf.nn.softmax)
        model = tf.keras.models.Model(inputs=[lstm_input, cnn_input],
                                      outputs=[network_output])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        self.model = model

    def fit(self, input_data, labels, num_epochs, batch_size, learn_rate=0.01):
        input_data = np.array(input_data)
        cnn_input = np.reshape(input_data, (-1, time_steps, num_features, 1))
        lstm_input = np.reshape(input_data, (-1, time_steps, num_features))
        self.model.fit({'lstm_input': cnn_input, 'cnn_input': lstm_input},
                       {'network_output': labels},
                       epochs=num_epochs, batch_size=batch_size)

    def evaluate(self, test_data, test_labels):
        loss, accuracy = self.model.evaluate(test_data, test_labels)
        print("Model loss:", loss, ", Accuracy:", accuracy)
        return loss, accuracy
