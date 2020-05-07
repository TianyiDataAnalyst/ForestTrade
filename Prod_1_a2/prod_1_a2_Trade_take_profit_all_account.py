# -*- coding: utf-8 -*-
"""
Created on Mon May  4 08:54:48 2020

@author: gutia
"""

# for all account 
import time
import oandapyV20
import oandapyV20.definitions.instruments as definstruments
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
#import matplotlib.pyplot as plt
from ForestTrade.config import oanda_login as account
from ForestTrade.config import token
import oandapyV20.endpoints.trades as trades
from ForestTrade.config import var_prod_1
import datetime as dt
import re
#initiating API connection and defining trade parameters

client = oandapyV20.API(str(token.token),environment="practice")
account_ids = account.account_universe
account_id = []
#account_id = '101-002-9736246-007'
#trade_id= '896'

def current_unit(trade_id,account_id):
    df_open_trade= pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])
    df_open_trade['currentUnits'] = df_open_trade['currentUnits'].apply(pd.to_numeric)
    #df_open_trade['price'] = df_open_trade['price'].apply(pd.to_numeric)
    df_open_trade.index = df_open_trade['id']
    return df_open_trade['currentUnits'][trade_id]

def unrealizedPL_trade(trade_id,account_id):      
    df_open_trade= pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])
    df_open_trade['unrealizedPL'] = df_open_trade['unrealizedPL'].apply(pd.to_numeric)
    df_open_trade.index = df_open_trade['id']
    return df_open_trade['unrealizedPL'][trade_id]

def trade_close(trade_id, curr_unit,account_id):
        data = {
                "units": str(curr_unit),
                "tradeID": str(trade_id)
                }
        r = trades.TradeClose(accountID=account_id, tradeID=trade_id, data=data)
        client.request(r)
        
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
    CandlestickGranularity = (definstruments.CandlestickGranularity().definitions.keys())
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
    trade_time_level = ''
    trade_time_lapse_hr = time_threshold(trade_id,account_id)/60
    if trade_time_lapse_hr >3 and trade_time_lapse_hr <6:
        trade_time_level = '3-6'
    if trade_time_lapse_hr >6:
        trade_time_level = '>6'
    if trade_time_lapse_hr <3:
        trade_time_level = '<3'
    return trade_time_level

#Take profit of open trades in each account 
def main():
    unit = var_prod_1.NUM_SHARES_PER_TRADE
    pnl_value = var_prod_1.pnl_value    
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
                if trade_time_lapse_level(trade_id,account_id)=='<3' and pnl > pnl_value:  #and currentUnits>0
                    print("Close trade:")
                    print("ID:", trade_id, "pnl_value:", pnl)   
                    trade_close(trade_id, unit, account_id)
                if trade_time_lapse_level(trade_id,account_id)=='>6'and pnl > 0:
                    print("Close trade:")
                    print("ID:", trade_id, "pnl_value:", pnl)   
                    trade_close(trade_id, unit, account_id)


#exposure risk and position size management montecarlo tree search
starttime=time.time()
timeout = time.time() + (60*60*24*170)  # 60 seconds times 60 meaning the script will run for 1 hr
while time.time() <= timeout:
    try:
        print( "passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main()
        time.sleep(30 - ((time.time() - starttime) % 30.0)) # orignial 300=5 minute interval between each new execution
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()
