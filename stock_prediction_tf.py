#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:12:06 2021

@author: altair
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

df = pd.read_csv("NSE-TATAGLOBAL11.csv")
print(df.head())

# analyze data in proper date time

df["Date"] = pd.to_datetime(df.Date, format = "%Y-%m-%d")
df.index = df['Date']

plt.figure(figsize=(16,8))
plt.plot(df["Close"], label = "Close Price History")

#implementation

# linear regression

# setting index as date values
df['Date'] = pd.to_datetime(df.Date, format= '%Y-%m-%d')
df.index = df['Date']

# sorting
data = df.sort_index(ascending= True, axis= 0)
data['Date'] = pd.to_numeric(pd.to_datetime(data['Date']))

# creating a seperate dataset

new_data = pd.DataFrame(index= range(0, len(df)), columns= ['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
    
#new_data['mon_fri'] = 0
#for i in range(0,len(new_data)):
#    if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
#        new_data['mon_fri'][i] = 1
#    else:
 #       new_data['mon_fri'][i] = 0
from fastai.tabular.all import *
add_datepart(new_data, 'Date')
new_data.drop('Elapsed', axis= 1, inplace= True) # elapsed will be the time stamp
        
# split into train and validation

train = new_data[:987]
valid = new_data[987:]

x_train = train.drop('Close', axis= 1)
y_train = train['Close']
x_valid = valid.drop('Close', axis= 1)
y_valid = valid['Close']

# implement linear regression

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

# results
# make predictions and find the RMSE

preds = model.predict(x_valid)
rms = np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print('RMS:', rms)

# plot

valid['Predictions'] = preds
#valid['Predictions'] = preds
#valid.loc[:,'Predictions']= preds
valid.loc[:, 'Predictions'] = preds

valid.index = new_data[987:].index
train.index = new_data[:987].index

plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])

# knn

from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MonMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

#scaling data

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_ttrain_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid_scaled)

# using gridsearch to find the best parameter
params = {'n_neighbors' : [2, 3, 4, 5, 6, 7, 8, 9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

# fit the model and make predictions
model.fit(x_train, y_train)
preds = model.predict(x_valid)

#rmse
rms = np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print('RMS:', rms)

# plots
#valid['Predictions'] = 0
#valid['Predictions'] = preds
valid.loc[:,-1]= preds

plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])