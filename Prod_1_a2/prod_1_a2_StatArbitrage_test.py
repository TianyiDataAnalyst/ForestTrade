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
#from ForestTrade.Prod_1.prod_1_1_2_StatArbitrage_strategy import output_delta
import re
import oandapyV20.endpoints.trades as trades

def final_delta_projected():   
    for line in open("C:\\Users\\gutia\\Anaconda3\\ForestTrade\\log\\prod_1_a2_final_delta_projected.txt"):
        pass
    #print(line)    
    regex=re.findall(r'(?<=value:).*?(?=\s)', line)
    final_delta_projected = ' '.join(map(str, regex))
    final_delta_projected = float(final_delta_projected)
    return final_delta_projected

#
CandlestickGranularity = (definstruments.CandlestickGranularity().definitions.keys())

#initiating API connection and defining trade parameters
client = oandapyV20.API(str(token.token),environment="practice")
account_id = account.oanda_pratice_account_id_a2

TRADING_INSTRUMENT = var_prod_1.TRADING_INSTRUMENT
         
pos_size = var_prod_1.NUM_SHARES_PER_TRADE

StatArb_VALUE_FOR_BUY_ENTRY = var_prod_1.VALUE_FOR_BUY_ENTRY

StatArb_VALUE_FOR_SELL_ENTRY = var_prod_1.VALUE_FOR_SELL_ENTRY

def candles(instrument):
    params = {"count": 100,"granularity": list(CandlestickGranularity)[9]} #granularity is in 'M15'; it can be in seconds S5 - S30, minutes M1 - M30, hours H1 - H12, days D[18], weeks W or months M
    candles = instruments.InstrumentsCandles(instrument=instrument,params=params)
    client.request(candles)
    ohlc_dict = candles.response["candles"]
    ohlc = pd.DataFrame(ohlc_dict)
    ohlc_df = ohlc.mid.dropna().apply(pd.Series)
    ohlc_df["volume"] = ohlc["volume"]
    ohlc_df.index = ohlc["time"]
    ohlc_df = ohlc_df.apply(pd.to_numeric)
    return ohlc_df


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
    params = {"count": 20,"granularity": list(CandlestickGranularity)[13]} #granularity is in 'M5'; it can be in seconds S5 - S30, minutes M1 - M30[10], hours H1 - H12, 2M[5],4M[6] 5M[7],15M[9],H2[12]
    candles = instruments.InstrumentsCandles(instrument=instrument,params=params)
    client.request(candles)
    ohlc_dict = candles.response["candles"]
    ohlc = pd.DataFrame(ohlc_dict)
    ohlc_df = ohlc.mid.dropna().apply(pd.Series)
    ohlc_df["volume"] = ohlc["volume"]
    ohlc_df.index = ohlc["time"]
    ohlc_m15_df = ohlc_df.apply(pd.to_numeric)
    return ohlc_m15_df

def market_order(instrument,units,sl):
    #"""units can be positive or negative, stop loss (in pips) added/subtracted to price """  
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

#n=20
def ATR(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['h']-df['l'])
    df['H-PC']=abs(df['h']-df['c'].shift(1))
    df['L-PC']=abs(df['l']-df['c'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=True)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return round(3*df2['ATR'][-1],5)

"""
df = candles_h3('USD_CAD')
data_h3 = candles_h3('USD_CAD')
ATR(data_h3,120)
"""
#type(StatArb_VALUE_FOR_SELL_ENTRY)
def trade_signal():
    signal = ""
    #"function to generate signal"
    if float(final_delta_projected()) > StatArb_VALUE_FOR_BUY_ENTRY:
        signal = "Buy"
    if float(final_delta_projected()) < StatArb_VALUE_FOR_SELL_ENTRY:
        signal = "Sell"
    return signal

def postion_balanace(l_s):
    df_open_trade= pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])    
    df_open_trade['currentUnits'] = df_open_trade['currentUnits'].apply(pd.to_numeric)
    count_short_position = df_open_trade['id'][df_open_trade['currentUnits'] > 0.0].count()
    count_long_position = df_open_trade['id'][df_open_trade['currentUnits'] < 0.0].count()
    if l_s =="short":
        if count_short_position <20:
            balance_signal = "short_open"
        elif count_short_position ==20:
            balance_signal = "short_close"
    if l_s =="long":            
        if count_long_position <19:
            balance_signal = "long_open"
        elif count_long_position ==19:
            balance_signal = "long_close"
    return balance_signal
#postion_balanace(l_s="short")

def main():
    currency =TRADING_INSTRUMENT
    print("StatAtr_test script analyzing ",currency)
    signal = trade_signal()
    if signal == "Buy" and postion_balanace(l_s="long") == 'long_open':
        market_order(currency,pos_size,str(300000))
        print(account_id, "New long position initiated for ", currency, " final_delta_projected: ", final_delta_projected())

    elif signal == "Sell" and postion_balanace(l_s="short") == 'short_open':
        market_order(currency,-1*pos_size,str(300000))
        print(account_id, "New short position initiated for ", currency, " final_delta_projected: ", final_delta_projected())

    else:
        print(account_id, "not meet the trade critiers", " final_delta_projected: ", final_delta_projected())
                
starttime=time.time()
timeout = time.time() + (60*60*24*170)  # 60 seconds times 60 meaning the script will run for 1 hr
while time.time() <= timeout:
    try:
        print(account_id, "passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main()
        time.sleep(900 - ((time.time() - starttime) % 900.0)) # orignial 300=5 minute interval between each new execution
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()

