#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:30:15 2021

@author: altair
"""

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

df = web.DataReader('AAPL', data_source= 'yahoo', start= '2015-01-01', end= '2020-12-31')
print(df)
print(df.shape)

# visualize closing price

plt.figure(figsize=(16,8))
plt.title('Close Price History',  fontsize= 18)
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD', fontsize= 18)
plt.show() 

# create new dataframe with close column
data = df.filter(['Close'])

# convert the dataframe to a numpy array
dataset = data.values

# convert the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * 0.8)
print('\n training_data_len:',training_data_len) 

#scale the data
scaler = MinMaxScaler(feature_range= (0, 1))
scaled_data = scaler.fit_transform(dataset)
print('\nscaled_data', scaled_data)

# create the training data set, scaled training dataset
train_data = scaled_data[0:training_data_len, :]

#split the daa into x_train and y_train dataset
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i,0])
    if i <= 60:
        print(x_train)
        print(y_train)
        print()
        
# convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print('\nx_train reshape:',x_train.shape)

# build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences= True, input_shape= (x_train.shape[1],1)))
model.add(LSTM(50,return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))

# compile the model
model.compile(optimizer='adam', loss= 'mean_squared_error')

#train the model
model.fit(x_train, y_train, batch_size= 1, epochs= 1)

# create the testing data, create dataset x_test, y_test
test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset[training_data_len, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# convert the data to a numpy array
x_test = np.array(x_test)

# reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

# get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# rmse
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print('\nRMSE:', rmse)

# plot
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# visualize
plt.figure(figsize=(16,8))
plt.title('LSTM Model for Stock Predictions', fontsize=18)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close' , 'Predictions']])
plt.legend(['Train', 'Valid', 'Predictions'], loc= 'lower right')
plt.show()

# get the quote
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2010-01-01', end= '2020-12-31')
new_df = apple_quote.filter(['Close'])
last_60_days = new_df[-60:].values
# scale the data
last_60_days_scaled = scaler.transform(last_60_days)

X_test = []
Y_test = []
X_test.append(last_60_days_scaled)

# convert into numpy array
X_test = np.array(X_test)
# reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

# get the predicted scaled price
pred_price = model.predict(X_test)
# undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print('Pred_price:', pred_price)

# get the quote
apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2020-12-31', end= '2020-12-31')
print('Close price:', apple_quote2['Close'])

##########################################
# linear regression
############################################


