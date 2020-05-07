#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:27:05 2020

@author: tianyigu
"""

import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.definitions.instruments as definstruments
import pandas as pd
#import matplotlib.pyplot as plt
import oandapyV20.endpoints.orders as orders
import time
from ForestTrade.config import oanda_login as account
from ForestTrade.config import token
from ForestTrade.config import var_prod_1
from ForestTrade.file_directory import file_name
import matplotlib.pyplot as plt
import oandapyV20.endpoints.trades as trades

pd.set_option('display.max_columns', None)
#initiating API connection and defining trade parameters
client = oandapyV20.API(str(token.token),environment="practice")
account_id = account.oanda_pratice_account_id

CandlestickGranularity = (definstruments.CandlestickGranularity().definitions.keys())

def candles(instrument):
    params = {"count": 100,"granularity": list(CandlestickGranularity)[18]} #granularity is in 'M15'; it can be in seconds S5 - S30, minutes M1 - M30, hours H1 - H12, days D[18], weeks W or months M
    candles = instruments.InstrumentsCandles(instrument=instrument,params=params)
    client.request(candles)
    ohlc_dict = candles.response["candles"]
    ohlc = pd.DataFrame(ohlc_dict)
    ohlc_df = ohlc.mid.dropna().apply(pd.Series)
    ohlc_df["volume"] = ohlc["volume"]
    ohlc_df.index = ohlc["time"]
    ohlc_df = ohlc_df.apply(pd.to_numeric)
    return ohlc_df

def convert_currency():
    df = candles('AUD_CAD')
    #df.index = pd.to_datetime(df.index) # change datetime to date, NOTE if granularity is less than day please delete
    df = df.reset_index()
    df.columns= ['Date','Open','High','Low','Close','Volume']
    #df.Date = pd.to_datetime(df.Date,format='%d.%m.%Y %H:%M:%S.%f')
    #df['Date'] = df['Date'].dt.date
    df = df.set_index(df.Date)
    df = df[['Open','High','Low','Close','Volume']]
    df = df.drop_duplicates(keep=False)
    
    df_2 = candles('XAU_CAD')
    df_2 = df_2.reset_index()
    df_2.columns= ['Date','Open','High','Low','Close','Volume']
    #df.Date = pd.to_datetime(df.Date,format='%d.%m.%Y %H:%M:%S.%f')
    #df['Date'] = df['Date'].dt.date
    df_2 = df_2.set_index(df_2.Date)
    df_2 = df[['Open','High','Low','Close','Volume']]
    df_2 = df.drop_duplicates(keep=False)
   
    
   
    
   _usd = []
    _usd= pd.DataFrame(_usd)
    
    _usd['Open'] = round(1/df['Open'],5)
    _usd['High'] = round(1/df['High'],5)   
    _usd['Low'] = round(1/df['Low'],5)
    _usd['Close'] = round(1/df['Close'],5)
    _usd['Volume'] = df['Volume']
    return _usd


instrument = "AUD_CAD"
instrument = "XAU_CAD"            
"XAU_AUD" 
df_xaucad = candles("XAU_CAD")
df_xauaud = candles("XAU_AUD")
audcad = candles("AUD_CAD")

mergedDf = df_xaucad.merge(df_xauaud , left_index=True, right_index=True)
mergedDf['rate']= round(mergedDf['c_x']/mergedDf['c_y'],5)   
merged_audcad = audcad.merge(mergedDf , left_index=True, right_index=True)
diff_goldR = merged_audcad
diff_goldR['diff'] = diff_goldR['c'] - diff_goldR['rate'] 

    plt.subplot(211)
    plt.plot(diff_goldR.loc[:,['c','sma_fast','sma_slow']])
    plt.title('SMA Crossover & Stochastic')
    plt.legend(('close','sma_fast','sma_slow'),loc='upper left')
    
    plt.subplot(212)
    plt.plot(diff_goldR.loc[:,['K','D']])
    plt.hlines(y=0.0001,xmin=0,xmax=len(diff_goldR),linestyles='dashed')
    plt.hlines(y=-0.0001,xmin=0,xmax=len(diff_goldR),linestyles='dashed')

plt.subplot(211)
plt.plot(diff_goldR.loc[:,['c','diff']])


report = diff_goldR[['c','rate','diff']]

report.to_csv (r'C:\Users\gutia\Desktop\data_audcad_xauRate.csv', index = True, header=True)