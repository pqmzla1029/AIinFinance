from nsepy import get_history
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import warnings
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import import_functions as ifunc

# Import and cut out first 10 rows
company,date1,date2= ifunc.read_file()
filename=company+".csv"
df = ifunc.read_full(filename)
df = df.drop(df.index[0:10], axis = 0)
df = df.reset_index()
df = df.drop(['index'], axis = 1)
df_scaled = df

# Min Max Scaler because profs said so
max_=df_scaled[['Open','High','Low','Close']].max().max()
min_=df_scaled[['Open','High','Low','Close']].min().min()

scl=MinMaxScaler()

X1=(df_scaled[['Open','High','Low','Close']]-min_)/(max_-min_)
X2=scl.fit_transform(df_scaled[['Volume']].values.reshape(-1,1))
X1=np.array(X1)

df_scaled=df_scaled.assign(Open=X1[:,0])
df_scaled=df_scaled.assign(High=X1[:,1])
df_scaled=df_scaled.assign(Low=X1[:,2])
df_scaled=df_scaled.assign(Close=X1[:,3])
df_scaled=df_scaled.assign(Volume=X2[:,0])

x=df_scaled[['Open','High','Low','Close','Volume','MACD','RSI SMA', 'RSI EWMA', 'Daily Change','5 Day Change']]
y=df.iloc[:,5]

