# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:20:54 2020

@author: gutia
"""
from config import oanda_login as account
from config import var_prod_1
import oandapyV20
import oandapyV20.endpoints.trades as trades
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import time
import oandapyV20.endpoints.forexlabs as labs
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.definitions.instruments as definstruments
import oandapyV20.endpoints.pricing as pricing
import datetime as dt
import re

# =============================================================================
# pd.set_option('display.max_rows', 1000) 
# pd.set_option('display.width', None)   
# pd.set_option('display.max_colwidth', 1000)
# =============================================================================

#initiating API connection and defining trade parameters
CandlestickGranularity = (definstruments.CandlestickGranularity().definitions.keys())

#initiating API connection and defining trade parameters
token_path = "C:\\Oanda\\token.txt" # Windows system format: "C:\\Oanda\\token.txt"; "token.txt" in PyCharm; ios "/Users/tianyigu/Downloads/token.txt"
client = oandapyV20.API(access_token=open(token_path,'r').read(),environment="practice")
account_id = account.oanda_pratice_account_id

#Globle variable 
trade_instrument = var_prod_1.TRADING_INSTRUMENT
NUM_SHARES_PER_TRADE = var_prod_1.NUM_SHARES_PER_TRADE 

#defining strategy parameters
#pairs = ['AUD_USD','GBP_USD','USD_CAD','USD_CHF','EUR_USD','USD_JPY','NZD_USD'] #currency pairs to be included in the strategy
#pairs = ['EUR_JPY','USD_JPY','AUD_JPY','AUD_USD','AUD_NZD','NZD_USD']

#instrument = 'NZD_CAD'
def candles(instrument):
    params = {"count": 50,"granularity": list(CandlestickGranularity)[4]} #granularity is in 'M15'; it can be in seconds S5 - S30, minutes M1 - M30, hours H1 - H12, days D[18], weeks W or months M
    candles = instruments.InstrumentsCandles(instrument=instrument,params=params)
    client.request(candles)
    ohlc_dict = candles.response["candles"]
    ohlc = pd.DataFrame(ohlc_dict)
    ohlc_df = ohlc.mid.dropna().apply(pd.Series)
    ohlc_df["volume"] = ohlc["volume"]
    ohlc_df.index = ohlc["time"]
    ohlc_df = ohlc_df.apply(pd.to_numeric)
    return ohlc_df

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
df = candles('USD_CAD')
ATR(df,50)
"""

def sys_time():
    params = {"instruments": trade_instrument}
    account_id = account.oanda_pratice_account_id
    r = pricing.PricingInfo(accountID=account_id, params=params)
    rv = client.request(r)
    print(rv["time"])

# https://developer.oanda.com/rest-live/forex-labs/#spreads
def spread_check(): #average spread
    data = {
            "instrument": trade_instrument,
            "period": 1  #period: 3600 - 1 hour. Required Period of time in seconds to retrieve spread data for. Values not in the following list will be automatically adjusted to the nearest valid value.
            }
    r = labs.Spreads(params=data)
    client.request(r)
    return r.response['avg'][-1][1]

#cost = NUM_SHARES_PER_TRADE*spread_check()*0.0001

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

#trade_id = '6450'
def order_createTime(trade_id):
    df_open_trade = pd.DataFrame(client.request(trades.OpenTrades(accountID=account_id))['trades'])
    df_stoploss_trade = pd.DataFrame(df_open_trade['stopLossOrder'])
    df_stoploss_trade = df_stoploss_trade.stopLossOrder.dropna().apply(pd.Series)
    df_stoploss_trade.index = df_stoploss_trade['tradeID']
    return df_stoploss_trade['createTime'].loc[trade_id] 
#current_createtime('6450-')
#df_stoploss_trade.to_csv (r'C:\Users\gutia\Desktop\export_dataframe.csv', index = False, header=True)


def format_datetime(date):
    conformed_timestamp = re.sub(r"[:]|([-](?!((\d{2}[:]\d{2})|(\d{4}))$))", '', date)
    x_day = dt.datetime(int(conformed_timestamp[0:4]),int(conformed_timestamp[4:6]),int(conformed_timestamp[6:8]), int(conformed_timestamp[9:11]), int(conformed_timestamp[11:13]))
    return x_day       

def time_threshold(trade_id):
    start_date = format_datetime(order_createTime(trade_id))
    end_date = format_datetime((candles(trade_instrument).index)[-1])
    return (end_date-start_date).seconds


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
        

#trade_close('6411','2000')
#trade_id ='6352'
def main():
    r = trades.OpenTrades(accountID=account_id)
    open_trades = client.request(r)['trades']
    trade_ids = []
    trade_id = []
    if len(open_trades)==0:
        print("no open trade: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))   
    for i in range(len(open_trades)):
        trade_ids.append(open_trades[i]['id'])
    trade_id = [i for i in trade_ids if i not in trade_ids]
    for trade_id in trade_ids:       
        unit = NUM_SHARES_PER_TRADE
        pnl = float(unrealizedPL_trade(trade_id))
        pnl_value = var_prod_1.pnl_value
        if pnl > pnl_value: # and time_threshold(trade_id) > 899:
            print("Close trade:")
            print("ID:", trade_id, "Unit:", unit)
            trade_close(trade_id, unit)
            f = open("C:\\Users\\gutia\\Documents\\GitHub\\ForestTrade\\Prod_1\\prod_1_pnl.txt","a+")
            f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) +", trade_id:" +str(trade_id)+',  pnl: '+str(pnl)+ '\n')
            f.close
 

starttime=time.time()
timeout = time.time() + (60*60*12)  # 60 seconds times 60 meaning the script will run for 1 hr
while time.time() <= timeout:
    try:
        print("Taking Profit script passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main()
        time.sleep(10*1 - ((time.time() - starttime) % 10.0)) # orignial 300=5 minute interval between each new execution
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()