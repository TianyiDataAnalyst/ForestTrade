# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 23:03:31 2020

@author: gutia
"""

import oandapyV20
#import matplotlib.pyplot as plt
import pandas as pd
from ForestTrade.config import token
from ForestTrade.config import oanda_login as account
import oandapyV20.endpoints.accounts as accounts
from ForestTrade.config import var_prod_1
import oandapyV20.endpoints.trades as trades

# =============================================================================
# pd.set_option('display.max_rows', 1000) 
# pd.set_option('display.width', None)   
# pd.set_option('display.max_colwidth', 1000)
# =============================================================================



#initiating API connection and defining trade parameters
client = oandapyV20.API(token.token,environment="practice")

account_ids = account.account_universe
account_id = []

#Globle variable 
trade_instrument = var_prod_1.TRADING_INSTRUMENT
NUM_SHARES_PER_TRADE = var_prod_1.NUM_SHARES_PER_TRADE 

pnl_value = 0

import oandapyV20.endpoints.positions as positions

def long_position_close(account_id):
    account_id = account_id
    instrument = var_prod_1.TRADING_INSTRUMENT
    data ={
           "longUnits": "ALL"
           }
    r = positions.PositionClose(accountID=accountID,instrument=instrument,data=data)
    client.request(r)
    return r.response

def short_position_close(account_id):
    account_id = account_id
    instrument = var_prod_1.TRADING_INSTRUMENT
    data ={
           "shortUnits": "ALL"
           }
    r = positions.PositionClose(accountID=accountID,instrument=instrument,data=data)
    client.request(r)
    return r.response


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
 
        

        
for account_id in account_ids:
    trade_id = []
    trade_ids = []
    acct_total_PL = [] 
    accountID=account_id
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
        if df_open_trade['id'][df_open_trade['currentUnits'] > 0.0].count() >0:
            long_position_close(account_id)
            print ('all long positions are closed')
        elif df_open_trade['id'][df_open_trade['currentUnits'] < 0.0].count() >0:
            short_position_close(account_id)
            print ('all short positions are closed')
    print('===============')