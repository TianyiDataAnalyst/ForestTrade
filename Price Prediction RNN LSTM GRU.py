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

df.columns= ['open','high','low','close','volume','symbol']

df.info()
df.head()

# number of different stocks
print('\nnumber of different stocks: ', len(list(set(df.symbol))))
print(list(set(df.symbol))[:10])

# If there is NULL value in the symbol, show the data 
#df[df.symbol.isnull()]