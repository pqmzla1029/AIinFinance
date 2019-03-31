# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import import_functions as ifunc
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from random import shuffle

start = time.time()

start=1200 # How many days back the backtesting data set should start
end=200 # How many days back the backtesting data set should end, 0 would be today
window=30 # Size of lookback period in predicting subsequent day, for example window = 30 means we use the past 30 trading days to predict the 31st day
per=0.9 # Percent of data set to use as training, the rest will be used to test the model
epochs = 100
dropout = 0.2

sc = MinMaxScaler(feature_range = (0, 1))
sc1 = MinMaxScaler(feature_range = (0, 1))
# User defines which features to use for training
use_Close = 1
use_Adj_Close = 0
use_Open = 0
use_High = 0
use_Low = 0
use_Volume = 0
use_MACD = 0
use_Signal_Line = 0
use_MACD_Crossover = 0
use_RSI_EWMA = 0
use_Daily_Change = 0
use_5_Day_Change = 0

###### Defining the Columns that we need to work with #######

df1 = pd.read_csv('AAPL.csv',index_col="date",parse_dates=True)
# company,date1,date2= ifunc.read_file()
# filename=company+".csv"
# df = ifunc.read_full(filename)
df1 = df1.drop(df1.index[0:11], axis = 0)
#    df = df.reset_index()
#    df = df.drop(['index'], axis = 1)
df=df1[-1*start:-1*end]
df_scaled = df1[-1*start:-1*end]
all_data = df1[-1*start:-1*end] #all_data = df.copy()

print()
print('The following features were used as inputs to the LSTM network:')
# Modifys data set to include desired featuress
if use_Close == 0:
    df = df.drop(['Close'], axis = 1)
    df_scaled = df_scaled.drop(['Close'], axis = 1)
else:
    df_scaled[['Close']] = sc1.fit_transform(df_scaled[['Close']])
    print('Close')

if use_Adj_Close == 0:
    df = df.drop(['Adj Close'], axis = 1)
    df_scaled = df_scaled.drop(['Adj Close'], axis = 1)
else:
    df_scaled[['Adj Close']] = sc.fit_transform(df_scaled[['Adj Close']])
    print('Adj Close')

if use_Open == 0:
    df = df.drop(['Open'], axis = 1)
    df_scaled = df_scaled.drop(['Open'], axis = 1)
else:
    df_scaled[['Open']] = sc.fit_transform(df_scaled[['Open']])
    print('Open')

if use_High == 0:
    df = df.drop(['High'], axis = 1)
    df_scaled = df_scaled.drop(['High'], axis = 1)
else:
    df_scaled[['High']] = sc.fit_transform(df_scaled[['High']])
    print('High')

if use_Low == 0:
    df = df.drop(['Low'], axis = 1)
    df_scaled = df_scaled.drop(['Low'], axis = 1)
else:
    df_scaled[['Low']] = sc.fit_transform(df_scaled[['Low']])
    print('Low')

if use_Volume == 0:
    df = df.drop(['Volume'], axis = 1)
    df_scaled = df_scaled.drop(['Volume'], axis = 1)
else:
    df_scaled[['Volume']] = sc.fit_transform(df_scaled[['Volume']])
    print('Volume')

if use_MACD == 0:
    df = df.drop(['MACD'], axis = 1)
    df_scaled = df_scaled.drop(['MACD'], axis = 1)
else:
    print('MACD')

if use_Signal_Line == 0:
    df = df.drop(['Signal'], axis = 1)
    df_scaled = df_scaled.drop(['Signal'], axis = 1)
else:
    print('MACD Signal Line (9 day moving average)')

if use_MACD_Crossover == 0:
    df = df.drop(['Crossover'], axis = 1)
    df_scaled = df_scaled.drop(['Crossover'], axis = 1)
else:
    print('MACD Crossover')

if use_RSI_EWMA == 0:
    df = df.drop(['RSI EWMA'], axis = 1)
    df_scaled = df_scaled.drop(['RSI EWMA'], axis = 1)
else:
   # df_scaled[['RSI EWMA']] = sc.fit_transform(df_scaled[['RSI EWMA']])
    print('EWMA Relative Strength Indicator')

if use_Daily_Change == 0:
    df = df.drop(['Daily Change'], axis = 1)
    df_scaled = df_scaled.drop(['Daily Change'], axis = 1)
else:
    print('Daily Change')

if use_5_Day_Change == 0:
    df = df.drop(['5 Day Change'], axis = 1)
    df_scaled = df_scaled.drop(['5 Day Change'], axis = 1)
else:
    print('5 Day Change')

print()

training_set_scaled=df_scaled#[-1*start:-1*end]
training_set=df#[-1*start:-1*end]


#training_set_scaled, training_set, all_data = globalVariableDefine()

############ splitting the data into Train and test ###############

xdata= []
ydata = []
length=training_set_scaled.shape[0]-1
column_idx = training_set_scaled.columns.get_loc('Close')
training_set_scaled = np.array(training_set_scaled)
training_set = np.array(training_set)

for i in range(window, length):
    xdata.append(training_set_scaled[i-window:i,:])
    ydata.append(training_set_scaled[i+1, column_idx]) # we may need to add +1

xdata, ydata = np.array(xdata), np.array(ydata)


split=int(per*len(xdata))
x_test=xdata[split:,:]
y_test=ydata[split:]
test_targets=training_set[-len(y_test):,column_idx]

x_train=xdata[:split,:]
y_train=ydata[:split]

ind_list = [i for i in range(len(x_train))]
shuffle(ind_list)
x_train = x_train[ind_list,:,:]
y_train = y_train[ind_list]

#x_train, y_train, x_test, y_test, test_targets = processData(training_set, training_set_scaled)
####



regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
regressor.add(Dropout(dropout))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True,activation='relu'))
regressor.add(Dropout(dropout))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True,activation='tanh'))
regressor.add(Dropout(dropout))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50,activation='relu'))
regressor.add(Dropout(dropout))

# Adding the output layer
regressor.add(Dense(units = 1, input_shape=(None,5)))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

model=regressor
#####
#model = buildModel(x_train)

model.fit(x_train, y_train, epochs = epochs, batch_size = 32,shuffle=True)

##### getting the score and plotting the retrn #####

xt = x_test
#    xt = np.reshape(xt, (xt.shape[0], xt.shape[1], xt.shape[2]))

####### trading code #######

predicted_stock_price = model.predict(xt)
predicted_stock_price = sc1.inverse_transform(predicted_stock_price)

predicted_price = pd.DataFrame(predicted_stock_price)
prediction=np.where(predicted_price<predicted_price.shift(-1),1,-1)

real_prices = pd.DataFrame(sc1.inverse_transform(test_targets.reshape(-1, 1)))
real_prices.columns = ['Close']
real_prices_shifted=np.where(real_prices<real_prices.shift(-1),1,-1)

score=accuracy_score(real_prices_shifted, prediction)


predicted_price = np.array(predicted_price)
real_prices['predicted price']=predicted_price

real_prices['returns']=np.log(real_prices.Close/real_prices.Close.shift(1))# calculating the every day return
real_prices['returns']=real_prices['returns'].shift(-1) # bringing the return to the previous day
real_prices['prediction_signal']=prediction # prediction signal
real_prices['value_prediction']=real_prices.prediction_signal*real_prices.returns #value prediction return

predicted_model_return = real_prices.value_prediction.cumsum()
predicted_market_return = real_prices.returns.cumsum()

#plottingTime(predicted_model_return, predicted_market_return, real_prices)
#########


plt.figure(1)
plt.clf()
plt.plot(predicted_model_return)
plt.plot(predicted_market_return)
plt.legend(["LSTM Return", "Market Return"])

plt.figure(2)
plt.clf()
plt.plot(real_prices['Close'], color = 'red', label = 'Real Stock Price')
plt.plot(real_prices['predicted price'], color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')

plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


#####
#score = getScore(x_test, model, sc, test_targets)
end = time.time()

print()
print('Model Run Time: ', end- start)
print('Final Accuracy: ', score)
print()