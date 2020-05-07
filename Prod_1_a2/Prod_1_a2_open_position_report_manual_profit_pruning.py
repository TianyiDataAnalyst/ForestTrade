#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:27:05 2020

@author: tianyigu
"""

import oandapyV20
import pandas as pd
#import matplotlib.pyplot as plt
from ForestTrade.config import oanda_login as account
from ForestTrade.config import token
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.accounts as accounts
from ForestTrade.config import var_prod_1
import datetime as dt
import re

#initiating API connection and defining trade parameters
client = oandapyV20.API(str(token.token),environment="practice")
account_id = account.oanda_pratice_account_id_a2

#Display all columns
pd.set_option('display.max_columns', None)

#Number of open positions:
r = trades.OpenTrades(accountID=account_id)
open_trades = client.request(r)['trades']  
df_open_trade= pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])    
df_open_trade.index[-1]

df_open_trade['unrealizedPL'] = df_open_trade['unrealizedPL'].apply(pd.to_numeric)
df_open_trade['currentUnits'] = df_open_trade['currentUnits'].apply(pd.to_numeric)

report_a2=(df_open_trade.groupby('currentUnits').count())['id']

df_open_trade['id'][df_open_trade['currentUnits'] < 0.0].count()


#Account details

r = accounts.AccountDetails(account_id)
client.request(r)
print (r.response)
account_info = client.request(r)['account'] 
account_details = pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])  

#In an account, take profitable positions and make Long- Short in balance
r = trades.OpenTrades(accountID=account_id)
open_trades = client.request(r)['trades']
df_open_trade= pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades']) 


def account_list():
    client = oandapyV20.API(str(token.token),environment="practice")
    r = accounts.AccountList()
    client.request(r)
    return r.response

#account_id = '101-002-9736246-007'
#trade_id= '896'

def account_summary(account_id):
    client = oandapyV20.API(str(token.token),environment="practice")
    r = accounts.AccountSummary(accountID=account_id)
    client.request(r)
    return r.response
df_acct_unrealizedPL= pd.DataFrame(client.request(accounts.AccountSummary(accountID=account_id))) 
pl = df_acct_unrealizedPL['account'][25]  

        
def current_unit(trade_id, account_id):#'unrealizedPL'
    df_open_trade= pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])
    df_open_trade['currentUnits'] = df_open_trade['currentUnits'].apply(pd.to_numeric)
    #df_open_trade['price'] = df_open_trade['price'].apply(pd.to_numeric)
    df_open_trade.index = df_open_trade['id']
    return df_open_trade['currentUnits'][trade_id]

def unrealizedPL_trade(trade_id, account_id):      
    df_open_trade= pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])
    df_open_trade['unrealizedPL'] = df_open_trade['unrealizedPL'].apply(pd.to_numeric)
    df_open_trade.index = df_open_trade['id']
    return df_open_trade['unrealizedPL'][trade_id]


def openTime_trade(trade_id, account_id):#'unrealizedPL'
    df_open_trade= pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])
    #df_open_trade['openTime'] = df_open_trade['openTime'].apply(pd.to_datetime)
    df_open_trade.index = df_open_trade['id']
    return df_open_trade['openTime'][trade_id]

def format_datetime(date):
    conformed_timestamp = re.sub(r"[:]|([-](?!((\d{2}[:]\d{2})|(\d{4}))$))", '', date)
    x_day = dt.datetime(int(conformed_timestamp[0:4]),int(conformed_timestamp[4:6]),int(conformed_timestamp[6:8]), int(conformed_timestamp[9:11]), int(conformed_timestamp[11:13]))
    return x_day

def candles(instrument):
    params = {"count": 1,"granularity": list(CandlestickGranularity)[0]} #granularity is in 'M15'; it can be in seconds S5 - S30, minutes M1 - M30, hours H1 - H12, days D[18], weeks W or months M
    candles = instruments.InstrumentsCandles(instrument=instrument,params=params)
    client.request(candles)
    ohlc_dict = candles.response["candles"]
    ohlc = pd.DataFrame(ohlc_dict)
    ohlc_df = ohlc.mid.dropna().apply(pd.Series)
    ohlc_df["volume"] = ohlc["volume"]
    ohlc_df.index = ohlc["time"]
    ohlc_df = ohlc_df.apply(pd.to_numeric)
    return ohlc_df


def time_threshold(trade_id,account_id):
    TRADING_INSTRUMENT= var_prod_1.TRADING_INSTRUMENT
    start_date = format_datetime(openTime_trade(trade_id, account_id))
    end_date = format_datetime((candles(TRADING_INSTRUMENT).index)[-1])
    return (end_date-start_date).seconds

def trade_time_lapse_level(trade_id,account_id):
    trade_time_lapse_hr = time_threshold(trade_id,account_id)/60
    if trade_time_lapse_hr >3 and trade_time_lapse_hr <6:
        trade_time_level = '3-6'
    if trade_time_lapse_hr >6:
        trade_time_level = '>6'
    if trade_time_lapse_hr <3:
        trade_time_level = '<3'
    return trade_time_level

def trade_close(trade_ID, curr_unit, account_id): #OANDA REST-V20 API Documentation,Page 67
        account_id= account_id
        data = {
                "units": str(curr_unit),
                "tradeID": str(trade_ID)
                }
        r = trades.TradeClose(accountID=account_id, tradeID=trade_ID, data=data)
        client.request(r)

#single account


#all accounts
def main():
    unit = var_prod_1.NUM_SHARES_PER_TRADE
    pnl_value = 0   
    for account_id in account_ids:
        print(account_id)        
        r = trades.OpenTrades(accountID=account_id)
        client = oandapyV20.API(str(token.token),environment="practice")
        open_trades = client.request(r)['trades']
        print(len(open_trades))
        trade_id = []
        trade_ids = []
        if len(open_trades)==0:
            continue
        if len(open_trades)>0:
            for i in range(len(open_trades)):
                trade_ids.append(open_trades[i]['id'])
            trade_id = [i for i in trade_ids if i not in trade_ids]
            for trade_id in trade_ids:       
                pnl = float(unrealizedPL_trade(trade_id,account_id))
                currentUnits = float(current_unit(trade_id,account_id))
                if pnl > pnl_value:  #and currentUnits>0
                    print("Close trade:")
                    print("ID:", trade_id, "pnl_value:", pnl)   
                    trade_close(trade_id, unit, account_id)
        
#check the numbers of open trades in each account 
#Number of open positions:
account_ids = account.account_universe
account_id = []

for account_id in account_ids:
    trade_id = []
    trade_ids = []
    r = trades.OpenTrades(accountID=account_id)
    client = oandapyV20.API(str(token.token),environment="practice")
    open_trades = client.request(r)['trades']
    r = trades.OpenTrades(accountID=account_id)
    df_open_trade= pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])    
    print(account_id)
    print('number of open positions: ', len(open_trades))
    if len(open_trades) > 0:
        df_acct_unrealizedPL= pd.DataFrame(client.request(accounts.AccountSummary(accountID=account_id))) 
        pl = df_acct_unrealizedPL['account'][25] 
        df_open_trade['currentUnits'] = df_open_trade['currentUnits'].apply(pd.to_numeric)
        print('number of + PnL positions: ', df_open_trade['id'][df_open_trade['currentUnits'] > 0.0].count())
        print('number of - PnL positions: ', df_open_trade['id'][df_open_trade['currentUnits'] < 0.0].count())
        print ('profitable positions: ', df_open_trade['id'][df_open_trade['unrealizedPL'].apply(pd.to_numeric) > 0.0].count())
        print ('total pl', pl)
    print('===============')