# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 10:39:33 2020

@author: gutia
"""

#https://stackoverflow.com/questions/1831410/python-time-comparison

import datetime
def todayAt (hr, min=0, sec=0, micros=0):
   now = datetime.datetime.now()
   return now.replace(hour=hr, minute=min, second=sec, microsecond=micros)    

# Usage demo1:
print (todayAt (17), todayAt (17, 15))

# Usage demo2:    
timeNow = datetime.datetime.now()
if timeNow < todayAt (13):
   print ("Too Early")
   
#=============================================================================
#  iso time     
import datetime
import dateutil.parser
import pytz

insertion_date = dateutil.parser.parse('2018-03-13T17:22:20.065Z')
diffretiation = pytz.utc.localize(datetime.datetime.utcnow()) - insertion_date


print (diffretiation) 
print (insertion_date)

if diffretiation.days>30:
    print ("The insertion date is older than 30 days")

else:
    print ("The insertion date is not older than 30 days")
#The insertion date is older than 30 days
#=============================================================================    
#=============================================================================
# Explore oanda iso time   
#=============================================================================
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.definitions.instruments as definstruments
import pandas as pd
import datetime as dt
import dateutil.parser
import pytz

#import matplotlib.pyplot as plt

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
    params = {"count": 250,"granularity": list(CandlestickGranularity)[0]} #granularity is in 'M15'; it can be in seconds S5 - S30, minutes M1 - M30, hours H1 - H12, days D, weeks W or months M
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

df= (candles('EUR_USD').index)[-1]
yourdate = dateutil.parser.parse(df)

insertion_date = yourdate
diffretiation = pytz.utc.localize(datetime.datetime.utcnow()) - insertion_date
print (diffretiation) 
print (insertion_date)

if diffretiation.days>30:
    print ("The insertion date is older than 30 days")
else:
    print ("The insertion date is not older than 30 days")

#display weekday
yourdate.strftime("%A %d. %B %Y")
#count business days
dt.date(dateutil.parser.parse((candles('EUR_USD').index)[-1]))

# N days ago

from datetime import datetime, timedelta

N = 2

date_N_days_ago = datetime.now() - timedelta(days=N)
print(date_N_days_ago)


import datetime as dt
import numpy as np
import re
# This regex removes all colons and all
# dashes EXCEPT for the dash indicating + or - utc offset for the timezone
last_row_date = (candles('EUR_USD').index)[-1]
first_row_date = (candles('EUR_USD').index)[1]


#date = '2020-04-01T19:01:00.000000000Z'
def format_date(date):
    conformed_timestamp = re.sub(r"[:]|([-](?!((\d{2}[:]\d{2})|(\d{4}))$))", '', date)
    x_day = dt.date(int(conformed_timestamp[0:4]),int(conformed_timestamp[4:6]),int(conformed_timestamp[6:8])+1)
    return x_day


def format_datetime(date):
    conformed_timestamp = re.sub(r"[:]|([-](?!((\d{2}[:]\d{2})|(\d{4}))$))", '', date)
    x_day = dt.datetime(int(conformed_timestamp[0:4]),int(conformed_timestamp[4:6]),int(conformed_timestamp[6:8]), int(conformed_timestamp[9:11]), int(conformed_timestamp[11:13]))
    return x_day    
end_date = format_datetime(last_row_date)
start_date = format_datetime(first_row_date)
#(end_date-start_date).days
#(end_date-start_date).microseconds
(end_date-start_date).seconds>90

pd.to_datetime(start_date).tz_localize('US/Eastern')
#dateutil.parser.parse((candles('EUR_USD').index)[-1])
end_day = dt.date.today()
#datetime.datetime.now().date()
business_days_diff = np.busday_count(start_date, end_date)
if business_days_diff>=1:
    print ("1 day")
else:
    print (business_days_diff)

def date_diff(start_date, end_date):
    start_timestamp = re.sub(r"[:]|([-](?!((\d{2}[:]\d{2})|(\d{4}))$))", '', start_date)
    end_timestamp = re.sub(r"[:]|([-](?!((\d{2}[:]\d{2})|(\d{4}))$))", '', end_date)
    cleaned_start = dt.date(int(start_timestamp[0:4]),int(start_timestamp[4:6]),int(start_timestamp[6:8]))
    cleaned_end = dt.date(int(end_timestamp[0:4]),int(end_timestamp[4:6]),int(end_timestamp[6:8]))
    busi_days_diff = np.busday_count(cleaned_start, cleaned_end)
    return(busi_days_diff)
(date_diff('2020-03-13', '2020-03-15'))>=1



#Generic
import oandapyV20.contrib.generic as generic
s = "M5"
generic.granularity_to_time(s)
datetime.datetime(2017, 6, 15, 4, 0)
d = generic.secs2time(1497499200)
d.strftime("%Y%m%d-%H:%M:%S")


