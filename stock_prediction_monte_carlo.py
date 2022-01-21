#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 00:31:57 2021

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

returns = np.log(1+df['Adj Close'].pct_change())

mu, sigma = returns.mean(), returns.std()
print('Random number:', np.random.normal(mu,sigma))
sim_rets = np.random.normal(mu,sigma, 1800)
initial = df['Adj Close'].iloc[-1]
sim_prices = initial * (sim_rets + 1).cumprod()
plt.plot(sim_prices)
plt.show()

for i in range(1000):
    sim_rets = np.random.normal(mu,sigma, 1800)
    initial = df['Adj Close'].iloc[-1]
    sim_prices = initial * (sim_rets + 1).cumprod()
    plt.axhline(initial,c='k')
    plt.plot(sim_prices)
    #plt.show()
    