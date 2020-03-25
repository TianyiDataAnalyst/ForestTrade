#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:27:05 2020

@author: tianyigu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:36:39 2020

@author: gutia
"""

# =============================================================================
# import pandas as pd
# from pandas_datareader import data
# =============================================================================

# Fetch daily data for 4 years
# =============================================================================
# SYMBOL='GOOG'
# start_date = '2014-01-01'
# end_date = '2018-01-01'
# SRC_DATA_FILENAME=SYMBOL + '_data.pkl'
# 
# try:
#  data = pd.read_pickle(SRC_DATA_FILENAME)
# except FileNotFoundError:
#  data = data.DataReader(SYMBOL, 'yahoo', start_date, end_date)
#  data.to_pickle(SRC_DATA_FILENAME)
# 
# =============================================================================
#data.head(4)
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.definitions.instruments as definstruments
import pandas as pd
#import matplotlib.pyplot as plt
import statistics as stats
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.orders as orders
import statsmodels.api as sm
import numpy as np
import time
#
CandlestickGranularity = (definstruments.CandlestickGranularity().definitions.keys())

#initiating API connection and defining trade parameters
token_path = "C:\\Oanda\\token.txt" # Windows system format: "C:\\Oanda\\token.txt"; "token.txt" in PyCharm; ios "/Users/tianyigu/Downloads/token.txt"
client = oandapyV20.API(access_token=open(token_path,'r').read(),environment="practice")
account_id = "101-002-9736246-001"

#defining strategy parameters
pairs = ['AUD_USD','GBP_USD','USD_CAD','USD_CHF','EUR_USD','USD_JPY','NZD_USD'] #currency pairs to be included in the strategy
#pairs = ['EUR_JPY','USD_JPY','AUD_JPY','AUD_USD','AUD_NZD','NZD_USD']

pos_size = 2000 

final_delta_projected_path = "C:\\Oanda\\Tradebot\\final_delta_projected.txt"
final_delta_projected = open(final_delta_projected_path,'r').read()

StatArb_VALUE_FOR_BUY_ENTRY = 0.01

StatArb_VALUE_FOR_SELL_ENTRY =0.01

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



def candles_h3(instrument):
    params = {"count": 200,"granularity": list(CandlestickGranularity)[13]} #granularity is in 'M5'; it can be in seconds S5 - S30, minutes M1 - M30[10], hours H1 - H12, 2M[5],4M[6] 5M[7],15M[9],H2[12]
    candles = instruments.InstrumentsCandles(instrument=instrument,params=params)
    client.request(candles)
    ohlc_dict = candles.response["candles"]
    ohlc = pd.DataFrame(ohlc_dict)
    ohlc_df = ohlc.mid.dropna().apply(pd.Series)
    ohlc_df["volume"] = ohlc["volume"]
    ohlc_df.index = ohlc["time"]
    ohlc_m15_df = ohlc_df.apply(pd.to_numeric)
    return ohlc_m15_df

#candles_m5('EUR_USD')['l'].tail(1)
#candles_h3('EUR_USD')['l'].tail(1) #3hours lowest point

def stochastic(df,a,b,c):
    #"function to calculate stochastic"
    df['k']=((df['c'] - df['l'].rolling(a).min())/(df['h'].rolling(a).max()-df['l'].rolling(a).min()))*100
    df['K']=df['k'].rolling(b).mean() 
    df['D']=df['K'].rolling(c).mean()
    return df

def SMA(df,a,b):
    #"function to calculate stochastic"
    df['sma_fast']=df['c'].rolling(a).mean() 
    df['sma_slow']=df['c'].rolling(b).mean() 
    return df

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
"""
df = candles('EUR_USD')
df["slope"] = slope(data["c"],5)
df["slope"].tail(1)

data.iloc[:,[3,5]].plot(subplots=True, layout = (2,1))
"""

def market_order(instrument,units,sl):
    #"""units can be positive or negative, stop loss (in pips) added/subtracted to price """  
    account_id = "101-002-9736246-001"
    data = {
            "order": {
            "price": "",
            "stopLossOnFill": {
            "trailingStopLossOnFill": "GTC",
            "distance": str(sl)
                              },
            "timeInForce": "FOK",
            "instrument": str(instrument),
            "units": str(units),
            "type": "MARKET",
            "positionFill": "DEFAULT"
                    }
            }
    r = orders.OrderCreate(accountID=account_id, data=data)
    client.request(r)
#market_order("USD_CAD","100","1.43")

def ATR(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['h']-df['l'])
    df['H-PC']=abs(df['h']-df['c'].shift(1))
    df['L-PC']=abs(df['l']-df['c'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return round(3*df2["ATR"][-1],5)

"""
data_h3 = candles_h3('CAD_JPY')
ATR(data_h3,120)
"""
# df = candles('EUR_USD')

def trade_signal(df):
    signal = ""
    #"function to generate signal"
    if float(final_delta_projected) > StatArb_VALUE_FOR_BUY_ENTRY:
        signal = "Buy"
    if float(final_delta_projected) > StatArb_VALUE_FOR_SELL_ENTRY:
        signal = "Sell"
    return signal
  
def main():
    global pairs
    for currency in pairs:
        if currency != "USD_CAD":
            continue
        print("analyzing ",currency)
        data = candles(currency)
        data_h3 = candles_h3(currency)
        ohlc_df = stochastic(data,14,3,3)
        ohlc_df = SMA(ohlc_df,100,200)
        signal = trade_signal(ohlc_df)
        if signal == "Buy":
            market_order(currency,pos_size,str(ATR(data_h3,120)))
            print("New long position initiated for ", currency)
            f = open("C:\\Oanda\\Tradebot\\log.txt", "a+")
            f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "New long position initiated for " + currency + '\n' )
            f.write("passthrough at " )
            f.close()
        elif signal == "Sell":
            market_order(currency,-1*pos_size,str(ATR(data_h3,120)))
            print("New short position initiated for ", currency)
            f = open("C:\\Oanda\\Tradebot\\log.txt", "a+")
            f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "New long position initiated for " + currency + '\n' )
            f.write("passthrough at " )
            f.close()
        else:
            print(currency, "not meet the trade critiers")
                
# =============================================================================
# starttime=time.time()
# timeout = time.time() + (60*0)+60*1  # 60 seconds times 60 meaning the script will run for 1 hr
# while time.time() <= timeout:
#     try:
#         print("passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
#         main()
#         time.sleep(15 - ((time.time() - starttime) % 15.0)) # orignial 300=5 minute interval between each new execution
#     except KeyboardInterrupt:
#         print('\n\nKeyboard exception received. Exiting.')
#         exit()
# =============================================================================
