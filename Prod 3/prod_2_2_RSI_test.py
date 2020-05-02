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
from ForestTrade.config import oanda_login_prod_2 as account
from ForestTrade.config import token
from ForestTrade.config import var_prod_2
from ForestTrade.file_directory import file_name
#from ForestTrade.Prod_1.prod_1_1_2_StatArbitrage_strategy import output_delta
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.forexlabs as labs

def computeRSI (data, time_window):
    diff = data.diff(1).dropna()        # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi


#
CandlestickGranularity = (definstruments.CandlestickGranularity().definitions.keys())

#initiating API connection and defining trade parameters
client = oandapyV20.API(str(token.token),environment="practice")
account_id = account.oanda_pratice_account_id
         
pos_size = var_prod_2.NUM_SHARES_PER_TRADE

pairs = var_prod_2.pairs

# https://developer.oanda.com/rest-live/forex-labs/#spreads
# currency = 'AUD_USD'
def spread_check(currency): #average spread
    data = {
            "instrument": str(currency),
            "period": 1  #period: 3600 - 1 hour. Required Period of time in seconds to retrieve spread data for. Values not in the following list will be automatically adjusted to the nearest valid value.
            }
    r = labs.Spreads(params=data)
    client.request(r)
    return r.response['avg'][-1][1]

def candles(instrument):
    n= var_prod_2.time_interval
    params = {"count": 100,"granularity": list(CandlestickGranularity)[n]} #granularity is in 'M15'; it can be in seconds S5 - S30, minutes M1 - M30, hours H1 - H12, days D[18], weeks W or months M
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


def BollBnd(DF,n):
    "function to calculate Bollinger Band"
    df = DF.copy()
    df["MA"] = df['c'].rolling(n).mean()
    df["BB_up"] = df["MA"] + 2*df['c'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_dn"] = df["MA"] - 2*df['c'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_width"] = df["BB_up"] - df["BB_dn"]
    df.dropna(inplace=True)
    return df

#currency="USD_CAD"
def trade_signal(currency):
    signal = ""
    risk_distance = var_prod_2.risk_distance
    rsi_buy_entry = var_prod_2.rsi_buy_entry
    rsi_sell_entry = var_prod_2.rsi_sell_entry
    #"function to generate signal"
    if BollBnd(candles(currency),20)['BB_width'][-1] >float(risk_distance) and computeRSI(candles(currency)['c'], 14)[-1] < rsi_buy_entry:
        signal = "Buy"
    if BollBnd(candles(currency),20)['BB_width'][-1] >float(risk_distance) and computeRSI(candles(currency)['c'], 14)[-1] > rsi_sell_entry:
        signal = "Sell"
    return signal

#type(BollBnd(candles(currency),20)['BB_width'][-1] )

def main():
    global pairs
    r = trades.OpenTrades(accountID=account_id)
    open_trades = client.request(r)['trades']
    curr_ls = []
    for i in range(len(open_trades)):
        curr_ls.append(open_trades[i]['instrument'])
    pairs = [i for i in pairs if i not in curr_ls]
    for currency in pairs:
        print("analyzing ",currency)
        print("RSI script analyzing ",currency)
        data_h3 = candles_h3(currency)
        signal = trade_signal(currency)
        if signal == "Buy":
            market_order(currency,pos_size,str(ATR(data_h3,20)))
            print("New long position initiated for ", currency, "RSI: ", computeRSI(candles(currency)['c'], 14)[-1] )   
        elif signal == "Sell":
            market_order(currency,-1*pos_size,str(ATR(data_h3,20)))
            print("New short position initiated for ", currency, "RSI: ", computeRSI(candles(currency)['c'], 14)[-1])
        else:
            print(currency, "not meet the trade critiers", currency, "RSI: ", computeRSI(candles(currency)['c'], 14)[-1])
                

starttime=time.time()
timeout = time.time() + (60*60*24*170)  # 60 seconds times 60 meaning the script will run for 1 hr  60*60*24*170 = 14,688,000‬
while time.time() <= timeout:
    print("---> passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    main()
    time.sleep(45 - ((time.time() - starttime) % 45.0)) # orignial 300=5 minute interval between each new execution