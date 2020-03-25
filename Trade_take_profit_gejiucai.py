# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:20:54 2020

@author: gutia
"""

import oandapyV20
import oandapyV20.endpoints.trades as trades
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import time

#initiating API connection and defining trade parameters
token_path = "C:\\Oanda\\Tradebot\\token.txt"
client = oandapyV20.API(access_token=open(token_path,'r').read(),environment="practice")
account_id = "101-002-9736246-001"


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
        account_id= "101-002-9736246-001"
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
timeout = time.time() + (60*0)+60*10  # 60 seconds times 60 meaning the script will run for 1 hr
while time.time() <= timeout:
    try:
        print("passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main()
        time.sleep(15 - ((time.time() - starttime) % 15.0)) # orignial 300=5 minute interval between each new execution
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()
# =============================================================================
# main()
#         
# account_id = "101-002-9736246-001"
# trade_ID = "2087"
# data = {
#         "units": '100'
#         }
# r = trades.TradeClose(accountID=account_id, tradeID=trade_ID,data=data)
# client.request(r)      
#    
# def order_price_func(trade_id):
#     df_open_trade= pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])
#     df_open_trade['price'] = df_open_trade['price'].apply(pd.to_numeric)
#     df_open_trade.index = df_open_trade['id']
#     return df_open_trade.loc[trade_id,"price"]
# 
# '''
#     if df_open_trade.loc[trade_id,'take_profit_signal'] < 0:
#         take_profit_signal = "Sell"
#         return take_profit_signal
#     if df_open_trade.loc[trade_id,'currentUnits'] > 0:
#         take_profit_signal = "Buy"
#         return take_profit_signal
# '''
# #order_price_func('2087')
# 
# def main():
#     trade_id=""
#     r = trades.OpenTrades(accountID=account_id)
#     open_trades = client.request(r)['trades']
#     trade_id_ls = []
#     for i in range(len(open_trades)):
#         trade_id_ls.append(open_trades[i]['id'])
#         #print(trade_id_ls)        
#     trade_id = [i for i in trade_id if i not in trade_id_ls]
#     print(trade_id)
#     for trade_id in trade_id_ls:
#         price = target_price(trade_id).loc[trade_id,'target_price'] 
#         take_profit_order(trade_id, price)
#         print("setup take profit ",trade_id, "at", price)
#         
#         
# accountID = "101-002-9736246-001"
# data = {
#         "order": {
#                 "timeInForce": "GTC",
#                 "price": 1.34026,
#                 "type": "TAKE_PROFIT",
#                 "tradeID": "2084"
#                 }
#         }
# r = orders.OrderCreate(accountID, data=data)
# client.request(r)
# 
# df_open_trade = pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])
# 
# 
# # Check if Dataframe is empty using empty attribute
# if df_open_trade.empty == True:
#     print('DataFrame is empty')
# else:
#     df_open_trade['currentUnits'] = df_open_trade['currentUnits'].apply(pd.to_numeric)
#     df_open_trade['price'] = df_open_trade['price'].apply(pd.to_numeric)
#     df_open_trade['unrealizedPL']
# 
# 
# # Check if Dataframe is empty using empty attribute
# if dfObj.empty == True:
#     print('DataFrame is empty')
# else:
#     print('DataFrame is not empty')
# df_open_trade['currentUnits'] = df_open_trade['currentUnits'].apply(pd.to_numeric)
# df_open_trade['price'] = df_open_trade['price'].apply(pd.to_numeric)
# df_open_trade['unrealizedPL']
# 
# 'unrealizedPL'
# 
# jpy_array = df_open_trade['instrument'].array
# if any("JPY" in i for i in jpy_array):
#     print (1)
# 
# jpy_array
# df_open_trade['instrument'].array
# df_open_trade.index = df_open_trade['id']
# df_open_trade['target_price'] = np.where(df_open_trade['currentUnits'] <0, round(df_open_trade['price']*0.0009,5), round(df_open_trade['price']*1.0001,5))
# df_open_trade['target_price']
# 
# 
# import oandapyV20.definitions.trades as deftrades
# print (deftrades.TradePL.POSITIVE)
# c = deftrades.TradePL()
# print (c[c.POSITIVE])
# print (deftrades.TradePL().definitions[c.POSITIVE])
# 
# 
# 
# import oandapyV20.endpoints.positions as positions
# r = positions.PositionDetails(accountID=account_id, instrument="USD_CAD")
# client.request(r)
# print (r.response)
# 
# 
# =============================================================================
