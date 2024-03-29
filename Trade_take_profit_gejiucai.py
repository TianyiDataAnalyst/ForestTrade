# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:20:54 2020

@author: gutia
"""
from config import oanda_login as account 
import oandapyV20
import oandapyV20.endpoints.trades as trades
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import time
import oandapyV20.endpoints.forexlabs as labs
#initiating API connection and defining trade parameters
token_path = "C:\\Oanda\\Tradebot\\token.txt"
client = oandapyV20.API(access_token=open(token_path,'r').read(),environment="practice")
account_id = account.oanda_pratice_account_id

#Globle variable 
trade_instrument = "NZD_CAD"
NUM_SHARES_PER_TRADE = 2000 

def spread_check(): #average spread
    data = {
            "instrument": trade_instrument,
            "period": 1
            }
    r = labs.Spreads(params=data)
    client.request(r)
    return r.response['avg'][-1][1]


def target_price(trade_id):
    df_open_trade= pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])
    df_open_trade['currentUnits'] = df_open_trade['currentUnits'].apply(pd.to_numeric)
    df_open_trade['price'] = df_open_trade['price'].apply(pd.to_numeric)
    df_open_trade.index = df_open_trade['id']
    jpy_array = df_open_trade['instrument'][trade_id]
    if any("JPY" in i for i in jpy_array):
        df_open_trade['target_price'] = np.where(df_open_trade['currentUnits'] <0, round(df_open_trade['price']*0.0009,3), round(df_open_trade['price']*1.0001,3))
    else:
        df_open_trade['target_price'] = np.where(df_open_trade['currentUnits'] <0, round(df_open_trade['price']*0.0009,5), round(df_open_trade['price']*1.0001,5))
    return df_open_trade['target_price'][trade_id]

def current_unit(trade_id):#'unrealizedPL'
    df_open_trade= pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])
    df_open_trade['currentUnits'] = df_open_trade['currentUnits'].apply(pd.to_numeric)
    #df_open_trade['price'] = df_open_trade['price'].apply(pd.to_numeric)
    df_open_trade.index = df_open_trade['id']
    return df_open_trade['currentUnits'][trade_id]

def unrealizedPL_trade(trade_id):#'unrealizedPL'
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
        

#trade_close('4179', '2000')

def main():
    r = trades.OpenTrades(accountID=account_id)
    open_trades = client.request(r)['trades']
    trade_ids = []
    trade_id = []
    try:
        if len(open_trades)==0:
            print("no open trade: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    except:
        for i in range(len(open_trades)):
            trade_ids.append(open_trades[i]['id'])
        trade_id = [i for i in trade_ids if i not in trade_ids]
        for trade_id in trade_ids:
            print("Close trade:")
            unit = 2000
            pnl = float(unrealizedPL_trade(trade_id))
            pnl_value = 0
            if pnl > pnl_value:
                print("ID:", trade_id, "Unit:", unit)
                trade_close(trade_id, unit)
            else: 
                "0"
                
starttime=time.time()
timeout = time.time() + (60*60*12)  # 60 seconds times 60 meaning the script will run for 1 hr
while time.time() <= timeout:
    try:
        print("Taking Profit script passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main()
        time.sleep(5*1 - ((time.time() - starttime) % 5.0)) # orignial 300=5 minute interval between each new execution
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()