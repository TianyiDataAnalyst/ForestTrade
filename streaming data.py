# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 10:21:56 2020

@author: gutia
"""


import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import warnings
warnings.filterwarnings('ignore')

#initiating API connection and defining trade parameters
token_path = "C:\\Oanda\\token.txt"
client = oandapyV20.API(access_token=open(token_path,'r').read(),environment="practice")
account_id = "101-002-9736246-001"

params = {"instruments": "USD_JPY"}
account_id = account_id
r = pricing.PricingInfo(accountID=account_id, params=params)
i=0
while i <=20:
    rv = client.request(r)
    print("Time=",rv["time"])
    print("bid=",rv["prices"][0]["closeoutBid"])
    print("ask=",rv["prices"][0]["closeoutAsk"])
    print("*******************")
    i+=1
    
    
    



import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.definitions.instruments as definstruments
import pandas as pd
#import matplotlib.pyplot as plt
import time
from ForestTrade.config import oanda_login as account
from ForestTrade.config import token
from ForestTrade.config import var_prod_1
import re
import datetime as dt
import numpy as np
import oandapyV20.contrib.generic as generic


#from ForestTrade.Prod_1.prod_1_1_2_StatArbitrage_strategy import output_delta


def final_delta_projected():   
    for line in open("C:\\Users\\gutia\\Anaconda3\\ForestTrade\\log\\prod_1_4_final_delta_projected.txt"):
        pass
    #print(line)    
    regex=re.findall(r'(?<=value:).*?(?=\s)', line)
    final_delta_projected = ' '.join(map(str, regex))
    final_delta_projected = float(final_delta_projected)
    return final_delta_projected

#
CandlestickGranularity = (definstruments.CandlestickGranularity().definitions.keys())

#initiating API connection and defining trade parameters
client = oandapyV20.API(str(token.token),environment="practice")
account_id = account.oanda_pratice_account_id

TRADING_INSTRUMENT = var_prod_1.TRADING_INSTRUMENT
         
pos_size = var_prod_1.NUM_SHARES_PER_TRADE

StatArb_VALUE_FOR_BUY_ENTRY = var_prod_1.VALUE_FOR_BUY_ENTRY

StatArb_VALUE_FOR_SELL_ENTRY = var_prod_1.VALUE_FOR_SELL_ENTRY

def format_date(date):
    conformed_timestamp = re.sub(r"[:]|([-](?!((\d{2}[:]\d{2})|(\d{4}))$))", '', date)
    x_day = dt.date(int(conformed_timestamp[0:4]),int(conformed_timestamp[4:6]),int(conformed_timestamp[6:8])+1)
    return x_day

def candles_1(instrument):
    params = {"count": 5000,"granularity": list(CandlestickGranularity)[9]} #granularity is in 'M15'; it can be in seconds S5 - S30, minutes M1 - M30, hours H1 - H12, days D[18], weeks W or months M
    candles = instruments.InstrumentsCandles(instrument=instrument,params=params)
    client.request(candles)
    ohlc_dict = candles.response["candles"]
    ohlc = pd.DataFrame(ohlc_dict)
    ohlc_df = ohlc.mid.dropna().apply(pd.Series)
    ohlc_df["volume"] = ohlc["volume"]
    ohlc_df.index = ohlc["time"]
    ohlc_df = ohlc_df.apply(pd.to_numeric)
    return ohlc_df
data_output_1= candles_1(TRADING_INSTRUMENT)

data_output_1.to_csv (r'C:\Users\gutia\Desktop\NZD_CAD.csv', index = True, header=True)

data_1 = (data_output_1.index)[0]
s = "M15"
delta_sec = generic.granularity_to_time(s)*5000

sec_data = dt.datetime(2020,2,5,1,15).timestamp()-18000-delta_sec

d = generic.secs2time(sec_data)
d.strftime("%Y%m%d-%H:%M:%S")


print (list(CandlestickGranularity)[9])
15*5000/60/24
dt.timedelta(minutes=5000)
format_date(data_1)

def date_diff(start_date, end_date):
    start_timestamp = re.sub(r"[:]|([-](?!((\d{2}[:]\d{2})|(\d{4}))$))", '', start_date)
    end_timestamp = re.sub(r"[:]|([-](?!((\d{2}[:]\d{2})|(\d{4}))$))", '', end_date)
    cleaned_start = dt.date(int(start_timestamp[0:4]),int(start_timestamp[4:6]),int(start_timestamp[6:8]))
    cleaned_end = dt.date(int(end_timestamp[0:4]),int(end_timestamp[4:6]),int(end_timestamp[6:8]))
    busi_days_diff = np.busday_count(cleaned_start, cleaned_end)
    return(busi_days_diff)

dt_from = dt.datetime(2020,2,5,1,15)
dt_to = dt.datetime(2020,2,5,1,0)
dt_from_unix = time.mktime(dt.datetime(2016, 6, 28, 4, 30).timetuple())
dt_to_unix = time.mktime(dt.datetime(2019,6, 28, 4, 30).timetuple())
dt_from_unix - dt_to_unix


def candles_2(instrument):
    params = {"count": 5000,"to": dt_to_unix,"granularity": list(CandlestickGranularity)[9]} #granularity is in 'M15'; it can be in seconds S5 - S30, minutes M1 - M30, hours H1 - H12, days D[18], weeks W or months M
    candles = instruments.InstrumentsCandles(instrument=instrument,params=params)
    client.request(candles)
    ohlc_dict = candles.response["candles"]
    ohlc = pd.DataFrame(ohlc_dict)
    ohlc_df = ohlc.mid.dropna().apply(pd.Series)
    ohlc_df["volume"] = ohlc["volume"]
    ohlc_df.index = ohlc["time"]
    ohlc_df = ohlc_df.apply(pd.to_numeric)
    return ohlc_df

data_output_5= candles_2(TRADING_INSTRUMENT)
data_output_5.to_csv (r'C:\Users\gutia\Desktop\NZD_CAD_5.csv', index = True, header=True)

mergedDf_12 = pd.merge(data_output_1, data_output_2, left_index=True, right_index=True)

mergedDf_12.describe()