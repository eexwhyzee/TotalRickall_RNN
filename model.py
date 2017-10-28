from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

def lstm_model(seq_length, chars, n_neurons=784):

    model = Sequential()
    model.add(LSTM(n_neurons,
                   input_shape=(seq_length, len(chars)),
                   dropout=0.2,
                   return_sequences=True))
    model.add(LSTM(n_neurons))
    model.add(Dense(len(chars),
                    activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam')
    return model


