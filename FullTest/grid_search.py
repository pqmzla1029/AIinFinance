# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:13:01 2019

@author: johnp
"""

from math import sqrt
from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import import_functions as ifunc
import numpy as np

def model_configs(n_input, n_nodes, n_epochs, n_batch, n_diff, n_test_train_split):
	# create configs
	configs = list()
	for i in n_input:
		for j in n_nodes:
			for k in n_epochs:
				for l in n_batch:
					for m in n_diff:
						for n in n_test_train_split:
							cfg = [i, j, k, l, m, n]
							configs.append(cfg)
	print('Total configs: %d' % len(configs))
	return configs

def model_fit(data, config):

	n_input, n_nodes, n_epochs, n_batch, n_diff, n_test_train_split = config

	x_train, y_train, x_test, y_test = load_data(data, n_input, False, n_test_train_split)

	n_features = 1
	
	model = Sequential()
	model.add(LSTM(n_nodes, activation='relu', input_shape=(n_input, n_features)))
	model.add(Dense(n_nodes, activation='relu'))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	
	model.fit(x_train, y_train, epochs=n_epochs, batch_size=n_batch, verbose=0)
	
	return model, x_test, y_test

def load_data(data, seq_len, normalise_window, test_train_split):

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

filename, date1, date2 = ifunc.read_file()
filename = filename+".csv"
data = ifunc.read_full(filename)

data = data.iloc[:,6]

N_input = [50]
N_nodes = [50]
N_epochs = [50]
N_batch = [1]
N_diff = [12]
N_split = [0.9] #This is the test train split, as a percentage

cfg_list = model_configs(N_input, N_nodes, N_epochs, N_batch, N_diff, N_split)

output = [model_fit(data, cfg) for cfg in cfg_list]

