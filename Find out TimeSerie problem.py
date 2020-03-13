#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 08:52:38 2020

@author: tianyigu
"""



"""
Find out why Line 88 and Line 98 diliver different number of rows. they all run Candle function supposely getting the same number of rows.

"""


import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.definitions.instruments as definstruments
import pandas as pd
#import matplotlib.pyplot as plt

CandlestickGranularity = (definstruments.CandlestickGranularity().definitions.keys())

#initiating API connection and defining trade parameters
token_path = "token.txt"
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
SYMBOLS = ['AUD_USD','CAD_USD','NZD_USD','SPX500_USD']
#SYMBOLS = ['AUD_USD','GBP_USD','CAD_USD','CHF_USD','EUR_USD','JPY_USD','NZD_USD']
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
jpyusd['Open'].count()
#jpyusd = jpyusd.assign(symbol=pd.Series('JPY_USD', index=jpyusd.index))
#jpyusd_dict = jpyusd.groupby('symbol').apply(lambda dfg: dfg.drop('symbol', axis=1).to_dict(orient='list')).to_dict()

chfusd = convert_currency('USD_CHF')

#chfusd = chfusd.assign(symbol=pd.Series('CHF_USD', index=chfusd.index))
#chfusd_dict = chfusd.groupby('symbol').apply(lambda dfg: dfg.drop('symbol', axis=1).to_dict(orient='list')).to_dict()

audusd = clean_format('AUD_USD')
audusd['Open'].count()
#audusd = audusd.assign(symbol=pd.Series('AUD_USD', index=audusd.index))
#audusd_dict = audusd.groupby('symbol').apply(lambda dfg: dfg.drop('symbol', axis=1).to_dict(orient='list')).to_dict()

#['GBP_USD','CAD_USD','EUR_USD','NZD_USD']
cadusd = convert_currency('USD_CAD')

gbpusd = clean_format('GBP_USD')
eurusd = clean_format('EUR_USD')
nzdusd = clean_format('NZD_USD')
spxusd = clean_format('SPX500_USD')
