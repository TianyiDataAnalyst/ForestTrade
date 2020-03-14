# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:31:09 2020

@author: gutia
"""
import oandapyV20
import oandapyV20.endpoints.trades as trades
import pandas as pd
#import matplotlib.pyplot as plt


#initiating API connection and defining trade parameters
token_path = "C:\\Oanda\\Tradebot\\token.txt"
client = oandapyV20.API(access_token=open(token_path,'r').read(),environment="practice")
account_id = "101-002-9736246-001"


df_open_trade = pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])


# Check if Dataframe is empty using empty attribute
if df_open_trade.empty == True:
    print('DataFrame is empty')
else:
    df_open_trade['currentUnits'] = df_open_trade['currentUnits'].apply(pd.to_numeric)
    df_open_trade['price'] = df_open_trade['price'].apply(pd.to_numeric)
    df_open_trade['unrealizedPL']
