# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:17:47 2020

@author: gutia
"""
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.definitions.instruments as definstruments
import pandas as pd
import numpy as np
#import oandapyV20.endpoints.orders as orders
from ForestTrade.config import oanda_login as account
from ForestTrade.config import token
from ForestTrade.config import var_prod_backtest_1
import pandas_datareader.data as web
import yfinance as yf

TRADING_INSTRUMENT = var_prod_backtest_1.TRADING_INSTRUMENT
#
CandlestickGranularity = (definstruments.CandlestickGranularity().definitions.keys())

#initiating API connection and defining trade parameters

client = oandapyV20.API(str(token.token),environment="practice")
account_id = account.oanda_pratice_account_id

#defining strategy parameters
#pairs = ['AUD_USD','GBP_USD','USD_CAD','USD_CHF','EUR_USD','USD_JPY','NZD_USD'] #currency pairs to be included in the strategy
#pairs = ['EUR_JPY','USD_JPY','AUD_JPY','AUD_USD','AUD_NZD','NZD_USD']


#15*5000/60/24 = 52.08
def candles(instrument):
    n=13
    params = {"count": 5000,"granularity": list(CandlestickGranularity)[n]} #granularity is in 'M15'[9]; M2 is 【5】it can be in seconds S5 - S30, minutes M1 - M30, hours H1[11] - H12, days D[18], weeks W or months M
    candles = instruments.InstrumentsCandles(instrument=instrument,params=params)
    client.request(candles)
    ohlc_dict = candles.response["candles"]
    ohlc = pd.DataFrame(ohlc_dict)
    ohlc_df = ohlc.mid.dropna().apply(pd.Series)
    ohlc_df["volume"] = ohlc["volume"]
    ohlc_df.index = ohlc["time"]
    ohlc_df = ohlc_df.apply(pd.to_numeric)
    return ohlc_df



data = candles(TRADING_INSTRUMENT)

data.iloc[:,[3,5]].plot(subplots=True, layout = (2,1))

#https://blog.quantinsti.com/volatility-and-measures-of-risk-adjusted-return-based-on-volatility/

# Pull NIFTY data from Yahoo finance 
NIFTY = candles(TRADING_INSTRUMENT)
shape(NIFTY)
# Compute the logarithmic returns using the Closing price 
NIFTY['Log_Ret'] = np.log(NIFTY['c'] / NIFTY['c'].shift(1))

# Compute Volatility using the pandas rolling standard deviation function
NIFTY['Volatility'] = NIFTY['Log_Ret'].rolling(window=252).std() * np.sqrt(252)
print(NIFTY.tail(15))

# Plot the NIFTY Price series and the Volatility
NIFTY[['c', 'Volatility']].plot(subplots=True, color='blue',figsize=(8, 6))