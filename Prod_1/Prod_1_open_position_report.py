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


#initiating API connection and defining trade parameters
client = oandapyV20.API(str(token.token),environment="practice")
account_id = account.oanda_pratice_account_id

#Number of open positions:
r = trades.OpenTrades(accountID=account_id)
open_trades = client.request(r)['trades']  
df_open_trade= pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])    
df_open_trade.index[-1]

df_open_trade['unrealizedPL'] = df_open_trade['unrealizedPL'].apply(pd.to_numeric)
df_open_trade['currentUnits'] = df_open_trade['currentUnits'].apply(pd.to_numeric)
report_1=df_open_trade.groupby('currentUnits').count()

df_open_trade['id'][df_open_trade['unrealizedPL'] > 0.0].count()

# close profitable positions and make Long- Short in balance
take_profit_value = 0.0
trade_ids = []
trade_id = []
currentUnits = df_open_trade['currentUnits']
r = trades.OpenTrades(accountID=account_id)
open_trades = client.request(r)['trades']
 

def unrealizedPL_trade(trade_id):      
    df_open_trade= pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])
    df_open_trade['unrealizedPL'] = df_open_trade['unrealizedPL'].apply(pd.to_numeric)
    df_open_trade.index = df_open_trade['id']
    return df_open_trade['unrealizedPL'][trade_id]

def trade_close(trade_ID, curr_unit): #OANDA REST-V20 API Documentation,Page 67
        account_id= account.oanda_pratice_account_id
        data = {
                "units": str(curr_unit),
                "tradeID": str(trade_ID)
                }
        r = trades.TradeClose(accountID=account_id, tradeID=trade_ID, data=data)
        client.request(r)

for i in range(len(open_trades)):
    trade_ids.append(open_trades[i]['id'])
trade_id = [i for i in trade_ids if i not in trade_ids]
for trade_id in trade_ids:       
    unit = 2000
    pnl = float(unrealizedPL_trade(trade_id))
    pnl_value = take_profit_value
    if pnl > pnl_value: 
        print("Close trade:")
        print("ID:", trade_id, "pnl_value:", pnl)   
        trade_close(trade_id, unit)


CandlestickGranularity = (definstruments.CandlestickGranularity().definitions.keys())

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


def main():
    currency =TRADING_INSTRUMENT
    print("StatAtr_test script analyzing ",currency)
    data_h3 = candles_h3(currency)
    signal = trade_signal()
    if signal == "Buy":
        market_order(currency,pos_size,str(ATR(data_h3,20)))
        print("New long position initiated for ", currency, " final_delta_projected: ", final_delta_projected())
        f = open(file_name('log\\trade_log.txt'), "a+")
        f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + " New long position initiated for " + currency + '\n' )
        f.close()
    elif signal == "Sell":
        market_order(currency,-1*pos_size,str(ATR(data_h3,20)))
        print("New short position initiated for ", currency, " final_delta_projected: ", final_delta_projected())
        f = open(file_name('log\\trade_log.txt'), "a+")
        f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + " New short position initiated for " + currency + '\n' )
        f.close()
    else:
        print(currency, "not meet the trade critiers", " final_delta_projected: ", final_delta_projected())
                



    
    