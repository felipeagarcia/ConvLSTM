# ConvLSTM
Implementation of a ConvLSTM architecture in TensorFlow Keras.

## Description
This code implements a network proposed by Karim et. al. [1], wich consists on three convolutional layers and a LSTM layer receiving
processing the input data parallel, then, the outputs of each layer are concatenated and used by a softmax layer.

[1] Karim, Fazle, et al. "LSTM fully convolutional networks for time series classification." IEEE Access 6 (2018): 1662-1669.

## Files
data/ contains the dataset data, having 20 classes with 10 examples by class
src/ contains the source code
