# -*- coding: utf-8 -*-
"""
Created on Sat May  2 00:34:40 2020

@author: gutia
"""

# Fibonacci Levels considering original trend as upward move
# https://blog.quantinsti.com/fibonacci-retracement-trading-strategy-python/

import pandas as pd
from ta.volatility import BollingerBands
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.definitions.instruments as definstruments

import matplotlib.pyplot as plt

from ForestTrade.config import oanda_login as account
from ForestTrade.config import token
from ForestTrade.config import var_prod_2


#
CandlestickGranularity = (definstruments.CandlestickGranularity().definitions.keys())

#initiating API connection and defining trade parameters
client = oandapyV20.API(str(token.token),environment="practice")
account_id = account.oanda_pratice_account_id_4

TRADING_INSTRUMENT = var_prod_2.TRADING_INSTRUMENT
         
pos_size = var_prod_2.NUM_SHARES_PER_TRADE


def candles(instrument):
    n=7
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





# Load datas
df = candles(TRADING_INSTRUMENT)

#Plot price
fig, ax = plt.subplots()
ax.plot(df.c, color='black')


price_max = max(df['c'])
price_min = min(df['c'])

# Fibonacci Levels considering original trend as upward move
diff = price_max - price_min
#level1 = price_max - 0.236 * diff
level2 = price_max - 0.382 * diff
level3 = price_max - 0.618 * diff

print ("Level", "Price")
#Several studies have been done & documented with histograms to investigate if 0.618, 0.382, etc retracements actually come up more frequently then any other arbitrarily chosen ratios.
#https://www.quantopian.com/posts/fibonacci-retracement-algorithm-attempt-please-evaluate
#print ("0.236", level1)
print ("0.382", level2)
print ("0.618", level3)
print ("1 ", price_min)

ax.axhspan(level1, price_min, alpha=0.4, color='lightsalmon')
ax.axhspan(level2, level1, alpha=0.5, color='palegoldenrod')
ax.axhspan(level3, level2, alpha=0.5, color='palegreen')
ax.axhspan(price_max, level3, alpha=0.5, color='powderblue')

plt.ylabel("Price")
plt.xlabel("Date")
plt.legend(loc=2)
plt.show()


import numpy as np
t = np.arange(price_min, price_max, .00001)
s = df['c'].apply(pd.to_numeric)
plt.plot(t, s)
import numpy as np
t = np.arange(-1, 1, .00001)
s = np.sin(2 * np.pi * t)

plt.plot(t, s)
# Draw a thick red hline at y=0 that spans the xrange
plt.axhline(linewidth=8, color='#d62728')

# Draw a default hline at y=1 that spans the xrange
plt.axhline(y=1)

# Draw a default vline at x=1 that spans the yrange
plt.axvline(x=1)

# Draw a thick blue vline at x=0 that spans the upper quadrant of the yrange
plt.axvline(x=0, ymin=0.75, linewidth=8, color='#1f77b4')

# Draw a default hline at y=.5 that spans the middle half of the axes
plt.axhline(y=.5, xmin=0.25, xmax=0.75)

plt.axhspan(0.25, 0.75, facecolor='0.5', alpha=0.5)

plt.axvspan(1.25, 1.55, facecolor='#2ca02c', alpha=0.5)

plt.show()