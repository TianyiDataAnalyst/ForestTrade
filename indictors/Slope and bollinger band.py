# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:05:02 2020

@author: gutia
"""
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.definitions.instruments as definstruments
import pandas as pd
import numpy as np
import statsmodels.api as sm
#import oandapyV20.endpoints.orders as orders
from ForestTrade.config import oanda_login as account
from ForestTrade.config import token
from ForestTrade.config import var_prod_backtest_1


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

def candles_m1(instrument):
    n=4
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

def slope(ser,n):
    #"function to calculate the slope of n consecutive points on a plot"
    slopes = [i*0 for i in range(n-1)]
    for i in range(n,len(ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

def BollBnd(DF,n):
    "function to calculate Bollinger Band"
    df = DF.copy()
    df["MA"] = df['c'].rolling(n).mean()
    df["BB_up"] = df["MA"] + 2*df['c'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_dn"] = df["MA"] - 2*df['c'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_width"] = df["BB_up"] - df["BB_dn"]
    df.dropna(inplace=True)
    return df

# Visualizing Bollinger Band of the stocks for last 100 data points
df=BollBnd(candles(TRADING_INSTRUMENT),20).iloc[-100:,[3,-4,-3,-2]]
df.plot(title="Bollinger Band")

data = candles(TRADING_INSTRUMENT)
df["slope"] = slope(df["c"],2)
df["slope"].tail(1)
df.iloc[:,[0,-1].iloc[-100:].plot(subplots=True, layout = (2,1))

df_1=candles_m1(TRADING_INSTRUMENT)
df_1["slope"] = slope(df_1["c"],2)
df_1.iloc[:,[3,-1]].iloc[-100:].plot(subplots=True, layout = (2,1))