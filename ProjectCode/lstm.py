import datetime
import warnings
import numpy as np
import fix_yahoo_finance as yf
import pandas_datareader as pdr
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

warnings.filterwarnings("ignore")

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #print 'yo'
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        #plt.legend()
    plt.show()
	
def model_fit(data, config):
    n_input, n_nodes, n_epochs, n_batch, n_diff, n_test_train_split = config
    x_train, y_train, x_test, y_test = load_data(data, n_input, False, n_test_train_split)
    n_features = 1
    model = Sequential()
    model.add(LSTM(4, activation='tanh', input_shape=(x_train.shape[1], x_train.shape[2])))#n_input, n_features
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2*n_nodes, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3*n_nodes, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(4*n_nodes, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(5*n_nodes, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mse'])
    #filepath="LSTM-weights-best.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    print(y_train.shape)
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=n_batch, verbose=1,validation_split=0.2)#,callbacks=[checkpoint])
  #  model.load_weights("LSTM-weights-best.hdf5")
    
    return model, x_test, y_test, x_train, y_train

def load_data(data, seq_len, normalise_window, test_train_split):
    #f = open(filename, 'r').read()
    #data = f.split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(test_train_split * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    #x_train = train[:, :-1]
    #y_train = train[:, -1]
    #x_test = result[int(row):, :-1]
    #y_test = result[int(row):, -1]
    
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]


    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print ("Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

