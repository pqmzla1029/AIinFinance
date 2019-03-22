from nsepy import get_history
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

start = date(2015,1,1)
end= date.today()

data = get_history(symbol="SBIN", start=start, end=end)

max_=data[['Open','High','Low','Close']].max().max()
min_=data[['Open','High','Low','Close']].min().min()

scl=MinMaxScaler()

X1=(data[['Open','High','Low','Close']]-min_)/(max_-min_)
X2=scl.fit_transform(data[['Volume']].values.reshape(-1,1))
X1=np.array(X1)

data=data.assign(Open=X1[:,0])
data=data.assign(High=X1[:,1])
data=data.assign(Low=X1[:,2])
data=data.assign(Close=X1[:,3])
data=data.assign(Volume=X2[:,0])
data.tail()


X=data[['Open','High','Low','Close','Volume']]
y=data.Last.shift(-1)

timestep=1
X_list=[]
y_list=[]
for i in range(timestep,len(X)):
    X_list.append(np.array(X.iloc[i-timestep:i]))
    y_list.append(y.iloc[i:i+timestep].values)

test_size=60
X_train=np.array(X_list[:-test_size])
y_train=np.expand_dims(np.array(y_list[:-test_size]),axis=2)
X_test=np.array(X_list[-test_size:])
y_test=np.expand_dims(np.array(y_list[-test_size:]),axis=2)

from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
# Please note that the number of neurons used in LSTM model is lesser than those in the RNN model
neurons=200
act='tanh'
dropout_ratio=0.80

model=Sequential()
model.add(LSTM(4,input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences=True))
model.add(Dense(neurons,activation=act))
model.add(Dropout(dropout_ratio))
model.add(Dense(neurons*2,activation=act))
model.add(Dropout(dropout_ratio))
model.add(Dense(neurons*3,activation=act))
model.add(Dropout(dropout_ratio))
model.add(Dense(neurons*4,activation=act))
model.add(Dropout(dropout_ratio))
model.add(Dense(neurons*5,activation=act))
model.add(Dropout(dropout_ratio))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mse'])


filepath="LSTM-weights-best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')


model.summary()


model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=1, validation_split=0.2,callbacks=[checkpoint])

model.load_weights("LSTM-weights-best.hdf5")


predict_close=[]
predict_close = list(model.predict(X_test))


predicted_close=[]
for i in range(len(y_test)):
    predicted_close.append((predict_close[i][0][0]))
predicted_close


actual_close=[]
for i in range(len(y_test)):
    actual_close.append((y_test[i][0][0]))
actual_close


performance=pd.DataFrame([list(predicted_close),list(actual_close)])

performance=performance.T


performance.columns=['Predicted','Actual']
Spread=performance.Actual-performance.Predicted


performance.tail()


s=2
plt.figure(figsize=(15,7))
plt.plot(Spread)
plt.plot(Spread.expanding().mean()+s*Spread.expanding().std(),c='g')
plt.plot(Spread.expanding().mean()-s*Spread.expanding().std(),c='r')
plt.ylabel('Spread')
plt.xlabel('Test Data')
plt.show()

plt.figure(figsize=(15,7))
# Plot the predicted and actual prices for comparison
plt.plot(performance.Predicted.iloc[:-1],c='y')
plt.plot(performance.Actual,c='b')
# Plot the sell signlas wherever the Spread is above the upper standard deviation band
plt.scatter(performance.Actual[(Spread>Spread.expanding().mean()+s*Spread.expanding().std())].index,
            performance.Actual[(Spread>Spread.expanding().mean()+s*Spread.expanding().std())],c='r',s=50)
# Plot the buy signlas wherever the Spread is below the lower standard deviation band
plt.scatter(performance.Actual[(Spread<Spread.expanding().mean()-s*Spread.expanding().std())].index,
            performance.Actual[(Spread<Spread.expanding().mean()-s*Spread.expanding().std())],c='g',s=50)
plt.legend(['Predicted_Close','Actual_Close','Over Bought','Over Sold'])
plt.ylabel('SBI Price')
plt.xlabel('Test Data')
plt.show()