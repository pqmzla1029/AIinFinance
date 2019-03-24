
# coding: utf-8

# In[168]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime


# In[169]:


dataset = pd.read_csv('Google_Stock_Price_Train.csv',index_col="Date",parse_dates=True)


# In[170]:


dataset.head()


# In[171]:


dataset.isna().any()


# In[172]:


dataset.info()


# In[173]:


dataset['Open'].plot(figsize=(16,6))


# In[174]:


# convert column "a" of a DataFrame
dataset["Close"] = dataset["Close"].str.replace(',', '').astype(float)


# In[175]:


dataset["Volume"] = dataset["Volume"].str.replace(',', '').astype(float)


# In[176]:


# 7 day rolling mean
dataset.rolling(7).mean().head(20)


# In[177]:


dataset['Open'].plot(figsize=(16,6))
dataset.rolling(window=30).mean()['Close'].plot()


# In[178]:


dataset['Close: 30 Day Mean'] = dataset['Close'].rolling(window=30).mean()
dataset[['Close','Close: 30 Day Mean']].plot(figsize=(16,6))


# In[179]:


# Optional specify a minimum number of periods
dataset['Close'].expanding(min_periods=1).mean().plot(figsize=(16,6))


# In[180]:


training_set=dataset['Open']
training_set=pd.DataFrame(training_set)


# In[181]:


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


# In[182]:


# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[183]:


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[184]:


# Initialising the RNN
regressor = Sequential()


# In[185]:


# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))


# In[186]:


# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)


# In[227]:


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test4.csv',index_col="Date",parse_dates=True)


# In[228]:


real_stock_price = dataset_test.iloc[:, 1:2].values


# In[229]:


dataset_test.head()


# In[230]:


dataset_test.info()


# In[231]:


dataset_test["Volume"] = dataset_test["Volume"].str.replace(',', '').astype(float)


# In[232]:


test_set=dataset_test['Open']
test_set=pd.DataFrame(test_set)


# In[233]:


test_set.info()


# In[234]:


# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[235]:


predicted_stock_price=pd.DataFrame(predicted_stock_price)
predicted_stock_price.info()


# In[236]:



# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

