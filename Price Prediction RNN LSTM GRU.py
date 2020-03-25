# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 12:45:36 2020

@author: gutia
"""

import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf

# split data in 80%/10%/10% train/validation/test sets
valid_set_size_percentage = 10 
test_set_size_percentage = 10 

#display parent directory and working directory
print(os.path.dirname(os.getcwd())+':', os.listdir(os.path.dirname(os.getcwd())));
print(os.getcwd()+':', os.listdir(os.getcwd()));

import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.definitions.instruments as definstruments
import oandapyV20.endpoints.orders as orders
import pandas as pd
#import matplotlib.pyplot as plt
import statistics as stats
#Time analysis
import datetime as dt
import numpy as np
import re
#
CandlestickGranularity = (definstruments.CandlestickGranularity().definitions.keys())

#initiating API connection and defining trade parameters
token_path = "C:\\Oanda\\token.txt" # Windows system format: "C:\\Oanda\\token.txt"; "token.txt" in PyCharm; ios "/Users/tianyigu/Downloads/token.txt"
client = oandapyV20.API(access_token=open(token_path,'r').read(),environment="practice")
account_id = "101-002-9736246-001"

#defining strategy parameters
pairs = ['AUD_USD','GBP_USD','USD_CAD','USD_CHF','EUR_USD','USD_JPY','NZD_USD'] #currency pairs to be included in the strategy
#pairs = ['EUR_JPY','USD_JPY','AUD_JPY','AUD_USD','AUD_NZD','NZD_USD']

def candles(instrument):
    params = {"count": 1500,"granularity": list(CandlestickGranularity)[18]} #granularity is in 'M15'; it can be in seconds S5 - S30, minutes M1 - M30, hours H1 - H12, days D[18], weeks W or months M
    candles = instruments.InstrumentsCandles(instrument=instrument,params=params)
    client.request(candles)
    ohlc_dict = candles.response["candles"]
    ohlc = pd.DataFrame(ohlc_dict)
    ohlc_df = ohlc.mid.dropna().apply(pd.Series)
    ohlc_df["volume"] = ohlc["volume"]
    ohlc_df.index = ohlc["time"]
    ohlc_df = ohlc_df.apply(pd.to_numeric)
    return ohlc_df
#candles('USD_CAD')

# =============================================================================
# def reverse_base_curr(instrument, data):
#   data = candles(instrument)
#   usd_array = pairs
#     if any("USD_" in i for i in usd_array):
# =============================================================================
def convert_currency(instrument):
    df = candles(instrument)
    #df.index = pd.to_datetime(df.index) # change datetime to date, NOTE if granularity is less than day please delete
    df = df.reset_index()
    df.columns= ['Date','Open','High','Low','Close','Volume']
    #df.Date = pd.to_datetime(df.Date,format='%d.%m.%Y %H:%M:%S.%f')
    #df['Date'] = df['Date'].dt.date
    df = df.set_index(df.Date)
    df = df[['Open','High','Low','Close','Volume']]
    df = df.drop_duplicates(keep=False)
    
    _usd = []
    _usd= pd.DataFrame(_usd)
    
    _usd['Open'] = round(1/df['Open'],5)
    _usd['High'] = round(1/df['High'],5)   
    _usd['Low'] = round(1/df['Low'],5)
    _usd['Close'] = round(1/df['Close'],5)
    _usd['Volume'] = df['Volume']
    return _usd

TRADING_INSTRUMENT = 'CAD_USD'
#SYMBOLS = ['AUD_USD','CAD_USD','NZD_USD','SPX500_USD','AU200_AUD']
SYMBOLS = ['AUD_USD','GBP_USD','CAD_USD','CHF_USD','EUR_USD','JPY_USD','NZD_USD','SPX500_USD','AU200_AUD']
def clean_format(instrument):
    df = candles(instrument)
    #df.index = pd.to_datetime(df.index) # change datetime to date, NOTE if granularity is less than day please delete
    df = df.reset_index()
    df.columns= ['Date','Open','High','Low','Close','Volume']
    #df.Date = pd.to_datetime(df.Date,format='"%Y%m%d%H%M%S"')   # change datetime to date,
    #df['Date'] = df['Date'].dt.date   # change datetime to date,
    df = df.set_index(df.Date)
    df = df[['Open','High','Low','Close','Volume']]
    df = df.drop_duplicates(keep=False)
    return df



#dictionary structure:https://realpython.com/python-dicts/
jpyusd = convert_currency('USD_JPY')
#jpyusd['Open'].count()

chfusd = convert_currency('USD_CHF')

audusd = clean_format('AUD_USD')
#audusd['Open'].count()

#['GBP_USD','CAD_USD','EUR_USD','NZD_USD']
cadusd = convert_currency('USD_CAD')

gbpusd = clean_format('GBP_USD')
eurusd = clean_format('EUR_USD')
nzdusd = clean_format('NZD_USD')
spxusd = clean_format('SPX500_USD')
au200aud = clean_format('AU200_AUD')



#symbols_data = {'AUD_USD' : audusd, 'NZD_USD': nzdusd, 'CAD_USD': cadusd,'SPX500_USD': spxusd,'AU200_AUD': au200aud }
symbols_data = {  'AU200_AUD': au200aud, 'SPX500_USD': spxusd,'JPY_USD': jpyusd, 'CHF_USD': chfusd, 'AUD_USD' : audusd,
                'NZD_USD': nzdusd, 'EUR_USD': eurusd, 'GBP_USD': gbpusd, 'CAD_USD': cadusd}

#Merge all in one dataframe
jpyusd_list = jpyusd.assign(symbol=pd.Series('JPY_USD', index=jpyusd.index))
#jpyusd_dict = jpyusd_list.groupby('symbol').apply(lambda dfg: dfg.drop('symbol', axis=1).to_dict(orient='list')).to_dict()

chfusd_list = chfusd.assign(symbol=pd.Series('CHF_USD', index=chfusd.index))
#chfusd_dict = chfusd_list.groupby('symbol').apply(lambda dfg: dfg.drop('symbol', axis=1).to_dict(orient='list')).to_dict()

audusd_list = audusd.assign(symbol=pd.Series('AUD_USD', index=audusd.index))
#audusd_dict = audusd_list.groupby('symbol').apply(lambda dfg: dfg.drop('symbol', axis=1).to_dict(orient='list')).to_dict()

gbpusd_list = gbpusd.assign(symbol=pd.Series('GBP_USD', index=gbpusd.index))

eurusd_list = eurusd.assign(symbol=pd.Series('EUR_USD', index=eurusd.index))

nzdusd_list = nzdusd.assign(symbol=pd.Series('NZD_USD', index=nzdusd.index))

spxusd_list = spxusd.assign(symbol=pd.Series('SPX500_USD', index=spxusd.index))

au200aud_list = au200aud.assign(symbol=pd.Series('AU200_AUD', index=au200aud.index))

cadusd_list = cadusd.assign(symbol=pd.Series('CAD_USD', index=cadusd.index))

df = pd.concat([jpyusd_list, chfusd_list, audusd_list, gbpusd_list,eurusd_list, nzdusd_list, spxusd_list, au200aud_list, cadusd_list])

df = df.reset_index()
df.columns= ['date','open','high','low','close','volume','symbol']
df = df.set_index(df.date)
df= df[['open','high','low','close','volume','symbol']]
df = df.drop_duplicates(keep=False)
df.info()
df.head()

# number of different stocks
print('\nnumber of different stocks: ', len(list(set(df.symbol))))
print(list(set(df.symbol))[:10])

# If there is NULL value in the symbol, show the data 
#df[df.symbol.isnull()]


# 2. Analyze data

#df.tail()

df.describe()

#PLOT 
print("plot 1")
plot_name = 'CAD_USD'

plt.figure(figsize=(15, 5));
plt.subplot(1,2,1);
plt.plot(df[df.symbol == plot_name].open.values, color='red', label='open')
plt.plot(df[df.symbol == plot_name].close.values, color='green', label='close')
plt.plot(df[df.symbol == plot_name].low.values, color='blue', label='low')
plt.plot(df[df.symbol == plot_name].high.values, color='black', label='high')
plt.title('forex price')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')
#plt.show()

plt.subplot(1,2,2);
plt.plot(df[df.symbol == plot_name].volume.values, color='black', label='volume')
plt.title('forex volume')
plt.xlabel('time [days]')
plt.ylabel('volume')
plt.legend(loc='best');

# =============================================================================
# 3. Manipulate data
# 
# - choose a specific currency
# - drop feature: volume
# - normalize stock data
# - create train, validation and test data sets
# =============================================================================


# function for min-max normalization of stock
def normalize_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['close'] = min_max_scaler.fit_transform(df['close'].values.reshape(-1,1))
    return df

# function to create train, validation, test data given stock data and sequence length
def load_data(stock, seq_len):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len): 
        data.append(data_raw[index: index + seq_len])
    
    data = np.array(data);
    valid_set_size = int(np.round(valid_set_size_percentage/100*data.shape[0]));  
    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]));
    train_set_size = data.shape[0] - (valid_set_size + test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]
    y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]
    
    x_test = data[train_set_size+valid_set_size:,:-1,:]
    y_test = data[train_set_size+valid_set_size:,-1,:]
    
    return [x_train, y_train, x_valid, y_valid, x_test, y_test]

# choose one stock
df_stock = df[df.symbol == plot_name].copy()
df_stock.drop(['symbol'],1,inplace=True)
df_stock.drop(['volume'],1,inplace=True)

cols = list(df_stock.columns.values)
print('df_stock.columns.values = ', cols)

# normalize stock
df_stock_norm = df_stock.copy()
df_stock_norm = normalize_data(df_stock_norm)

# create train, test data
seq_len = 20 # choose sequence length
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df_stock_norm, seq_len)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid.shape = ',x_valid.shape)
print('y_valid.shape = ', y_valid.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ',y_test.shape)


# Visulize
print("Plot 2")
plt.figure(figsize=(15, 5));
plt.plot(df_stock_norm.open.values, color='red', label='open')
plt.plot(df_stock_norm.close.values, color='green', label='low')
plt.plot(df_stock_norm.low.values, color='blue', label='low')
plt.plot(df_stock_norm.high.values, color='black', label='high')
#plt.plot(df_stock_norm.volume.values, color='gray', label='volume')
plt.title('stock')
plt.xlabel('time [days]')
plt.ylabel('normalized price/volume')
plt.legend(loc='best')
plt.show()


# =============================================================================
# 4. Model and validate data 
# RNNs with basic, LSTM, GRU cells
# =============================================================================

## Basic Cell RNN in tensorflow
from tensorflow.python.framework import ops

        
index_in_epoch = 0;
perm_array  = np.arange(x_train.shape[0])
np.random.shuffle(perm_array)

# function to get the next batch
def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array   
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array) # shuffle permutation array
        start = 0 # start next epoch
        index_in_epoch = batch_size
        
    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]

# parameters
n_steps = seq_len-1 
n_inputs = 4 
n_neurons = 200 
n_outputs = 4
n_layers = 2
learning_rate = 0.001
batch_size = 50
n_epochs = 100 
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]

ops.reset_default_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

# use Basic RNN Cell
layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
          for layer in range(n_layers)]

# use Basic LSTM Cell 
#layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.elu)
#          for layer in range(n_layers)]

# use LSTM Cell with peephole connections
#layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons, 
#                                  activation=tf.nn.leaky_relu, use_peepholes = True)
#          for layer in range(n_layers)]

# use GRU cell
#layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu)
#          for layer in range(n_layers)]
                                                                     
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons]) 
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
outputs = outputs[:,n_steps-1,:] # keep only last output of sequence
                                              
loss = tf.reduce_mean(tf.square(outputs - y)) # loss function = mean squared error 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
training_op = optimizer.minimize(loss)
                                              
# run graph
with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    for iteration in range(int(n_epochs*train_set_size/batch_size)):
        x_batch, y_batch = get_next_batch(batch_size) # fetch the next training batch 
        sess.run(training_op, feed_dict={X: x_batch, y: y_batch}) 
        if iteration % int(5*train_set_size/batch_size) == 0:
            mse_train = loss.eval(feed_dict={X: x_train, y: y_train}) 
            mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid}) 
            print('%.2f epochs: MSE train/valid = %.6f/%.6f'%(
                iteration*batch_size/train_set_size, mse_train, mse_valid))

    y_train_pred = sess.run(outputs, feed_dict={X: x_train})
    y_valid_pred = sess.run(outputs, feed_dict={X: x_valid})
    y_test_pred = sess.run(outputs, feed_dict={X: x_test})
    












