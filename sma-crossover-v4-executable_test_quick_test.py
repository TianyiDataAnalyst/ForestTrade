# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 08:26:41 2020

@author: gutia
"""

import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.definitions.instruments as definstruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import pandas as pd
import numpy as np
import statsmodels.api as sm
#import matplotlib.pyplot as plt
import time

#
CandlestickGranularity = (definstruments.CandlestickGranularity().definitions.keys())

#initiating API connection and defining trade parameters
token_path = "C:\\Oanda\\Tradebot\\token.txt"
client = oandapyV20.API(access_token=open(token_path,'r').read(),environment="practice")
account_id = "101-002-9736246-001"

#defining strategy parameters
pairs = ['AUD_CAD','AUD_USD','CAD_CHF','CAD_HKD','CAD_SGD','EUR_AUD','EUR_CAD','EUR_GBP','EUR_USD','GBP_USD','NZD_CAD','USD_CAD','USD_CNH'] #currency pairs to be included in the strategy
#pairs = ['EUR_JPY','USD_JPY','AUD_JPY','AUD_USD','AUD_NZD','NZD_USD']
pos_size = 2000 #max capital allocated/position size for any currency pair
upward_dir = {}
dnward_dir = {}
for i in pairs:
    upward_dir[i] = False
    dnward_dir[i] = False

def candles(instrument):
    params = {"count": 250,"granularity": list(CandlestickGranularity)[4]} #granularity is in 1 minute[4] 'M15'; it can be in seconds S5 - S30, minutes M1 - M30, hours H1 - H12, days D, weeks W or months M
    candles = instruments.InstrumentsCandles(instrument=pairs[0],params=params)
    client.request(candles)
    ohlc_dict = candles.response["candles"]
    ohlc = pd.DataFrame(ohlc_dict)
    ohlc_df = ohlc.mid.dropna().apply(pd.Series)
    ohlc_df["volume"] = ohlc["volume"]
    ohlc_df.index = ohlc["time"]
    ohlc_df = ohlc_df.apply(pd.to_numeric)
    return ohlc_df
#candles('EUR_USD')


def candles_h3(instrument):
    params = {"count": 200,"granularity": list(CandlestickGranularity)[13]} #granularity is in 'M5'; it can be in seconds S5 - S30, minutes M1 - M30[10], hours H1 - H12, 2M[5],4M[6] 5M[7],15M[9],H2[12]
    candles = instruments.InstrumentsCandles(instrument=instrument,params=params)
    client.request(candles)
    ohlc_dict = candles.response["candles"]
    ohlc = pd.DataFrame(ohlc_dict)
    ohlc_df = ohlc.mid.dropna().apply(pd.Series)
    ohlc_df["volume"] = ohlc["volume"]
    ohlc_df.index = ohlc["time"]
    ohlc_m15_df = ohlc_df.apply(pd.to_numeric)
    return ohlc_m15_df

#candles_m5('EUR_USD')['l'].tail(1)
#candles_h3('EUR_USD')['l'].tail(1) #3hours lowest point

def stochastic(df,a,b,c):
    #"function to calculate stochastic"
    df['k']=((df['c'] - df['l'].rolling(a).min())/(df['h'].rolling(a).max()-df['l'].rolling(a).min()))*100
    df['K']=df['k'].rolling(b).mean() 
    df['D']=df['K'].rolling(c).mean()
    return df

def SMA(df,a,b):
    #"function to calculate stochastic"
    df['sma_fast']=df['c'].rolling(a).mean() 
    df['sma_slow']=df['c'].rolling(b).mean() 
    return df

def slope(ser,n):
    #"function to calculate the slope of n consecutive points on a plot"
    slopes = [i*0 for i in range(n-1)]
    for i in range(n,len(ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)
"""
df = candles('EUR_USD')
df["slope"] = slope(data["c"],5)
df["slope"].tail(1)

data.iloc[:,[3,5]].plot(subplots=True, layout = (2,1))
"""

def market_order(instrument,units,sl):
    """units can be positive or negative, stop loss (in pips) added/subtracted to price """  
    account_id = "101-002-9736246-001"
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

def ATR(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['h']-df['l'])
    df['H-PC']=abs(df['h']-df['c'].shift(1))
    df['L-PC']=abs(df['l']-df['c'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return round(3*df2["ATR"][-1],5)

"""
data_h3 = candles_h3('CAD_JPY')
ATR(data_h3,120)
"""
# df = candles('EUR_USD')

def trade_signal(df):
    signal = ""
    df["slope"] = slope(df["c"],5)
    "function to generate signal"
    if (df["slope"].tail(1)<-45).bool() == True:
        signal = "Buy"
    if (df["slope"].tail(1)>45).bool() == True:
        signal = "Sell"
    return signal
"""   
df = candles('EUR_USD')
df["slope"] = slope(data["c"],5)
df["slope"].tail(1)
trade_signal(df)
"""
def main():
    for currency in pairs:
        print("analyzing ",currency)
        data = candles(currency)
        #data_h3 = candles_h3(currency)
        #signal = trade_signal(data,currency)
        for currency in pairs:
                print("analyzing ",currency)
                data = candles(currency)
                ohlc_df = stochastic(data,14,3,3)
                ohlc_df = SMA(ohlc_df,100,200)
                signal = trade_signal(ohlc_df)
                if signal == "Buy":
                    market_order(currency,pos_size,str(ATR(data,120)))
                    print("New long position initiated for ", currency)
                    f = open("C:\\Oanda\\Tradebot\\log.txt", "a+")
                    f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "New long position initiated for " + currency + '\n' )
                    f.write("passthrough at " )
                    f.close()
                elif signal == "Sell":
                    market_order(currency,-1*pos_size,str(ATR(data,120)))
                    print("New short position initiated for ", currency)
                    f = open("C:\\Oanda\\Tradebot\\log.txt", "a+")
                    f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "New long position initiated for " + currency + '\n' )
                    f.write("passthrough at " )
                    f.close()
                else:
                    print(currency, "not meet the trade critiers")
                
starttime=time.time()
timeout = time.time() + (60*0)+60*1  # 60 seconds times 60 meaning the script will run for 1 hr
while time.time() <= timeout:
    try:
        print("passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main()
        time.sleep(15 - ((time.time() - starttime) % 15.0)) # orignial 300=5 minute interval between each new execution
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()