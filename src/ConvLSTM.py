import tensorflow as tf
import numpy


class ConvLSTM(Object):
    def __init__(self, num_classes, num_lstm_cells=128, num_lstm_layers=1,
                 cnn_filters=(3, 3), pool_size=(3, 3), num_cnn_cells=128,
                 num_cnn_layers=2):
        self.num_lstm_cells = num_lstm_cells
        self.num_lstm_layers = num_cnn_layers
        self.cnn_filters = cnn_filters
        self.pool_size = pool_size
        self.num_cnn_layers = num_cnn_layers
        self.num_classes = num_classes

    def create_cnn_model(self, input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(self.num_cnn_cells, self.cnn_filters,
                  input_shape=input_shape))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=self.pool_size))
        for layer in range(1, self.num_lstm_layers):
            model.add(tf.keras.layers.Conv2D(self.num_cnn_cells,
                      self.cnn_filters))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=self.pool_size))
        model.add
        return model

    def create_lstm_model(self):
        model = tf.keras.models.Sequential()
        return model

    def create_network(self, input_data):
        cnn = self.create_cnn_model(np.array(input_data).shape[1:])
        lstm = self.create_lstm_model()
