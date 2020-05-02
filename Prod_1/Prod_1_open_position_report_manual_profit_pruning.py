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
import oandapyV20.endpoints.positions as positions
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

report_1=(df_open_trade.groupby('currentUnits').count())['id']

df_open_trade['id'][df_open_trade['unrealizedPL'] > 0.0].count()


#Account details

r = accounts.AccountDetails(account_id)
client.request(r)
print (r.response)
account_info = client.request(r)['account'] 
account_details = pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])  

# close profitable positions and make Long- Short in balance
r = trades.OpenTrades(accountID=account_id)
open_trades = client.request(r)['trades']
df_open_trade= pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades']) 
 
take_profit_value = 7
trade_ids = []
trade_id = []



r = positions.PositionList(accountID=account_id)
position_list = client.request(r)['positions']


 
def current_unit(trade_id):#'unrealizedPL'
    df_open_trade= pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])
    df_open_trade['currentUnits'] = df_open_trade['currentUnits'].apply(pd.to_numeric)
    #df_open_trade['price'] = df_open_trade['price'].apply(pd.to_numeric)
    df_open_trade.index = df_open_trade['id']
    return df_open_trade['currentUnits'][trade_id]

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
    currentUnits = float(current_unit(trade_id))
    if pnl > pnl_value and currentUnits>0: 
        print("Close trade:")
        print("ID:", trade_id, "pnl_value:", pnl)   
        trade_close(trade_id, unit)