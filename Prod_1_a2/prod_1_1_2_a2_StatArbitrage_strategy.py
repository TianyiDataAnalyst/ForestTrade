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
import statistics as stats
import numpy as np
import time
from ForestTrade.config import var_prod_1
from ForestTrade.config import oanda_login as account
from ForestTrade.config import token
#
CandlestickGranularity = (definstruments.CandlestickGranularity().definitions.keys())

#initiating API connection and defining trade parameters
client = oandapyV20.API(token.token,environment="practice")
account_id = account.oanda_pratice_account_id

#defining strategy parameters
#pairs = ['AUD_USD','GBP_USD','USD_CAD','USD_CHF','EUR_USD','USD_JPY','NZD_USD'] #currency pairs to be included in the strategy
#pairs = ['EUR_JPY','USD_JPY','AUD_JPY','AUD_USD','AUD_NZD','NZD_USD']

pos_size = var_prod_1.NUM_SHARES_PER_TRADE

def candles(instrument):
    n=var_prod_1.time_interval
    params = {"count": 1500,"granularity": list(CandlestickGranularity)[n]} #granularity is in 'M15'[9]; it can be in seconds S5 - S30, minutes M1 - M30, hours H1 - H12, days D[18], weeks W or months M
    candles = instruments.InstrumentsCandles(instrument=instrument,params=params)
    client.request(candles)
    ohlc_dict = candles.response["candles"]
    ohlc = pd.DataFrame(ohlc_dict)
    ohlc_df = ohlc.mid.dropna().apply(pd.Series)
    ohlc_df["volume"] = ohlc["volume"]
    ohlc_df.index = ohlc["time"]
    ohlc_df = ohlc_df.apply(pd.to_numeric)
    return ohlc_df
#candles('USD_CAD')

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

# =============================================================================
# def reverse_base_curr(instrument, data):
#   data = candles(instrument)
#   usd_array = pairs
#     if any("USD_" in i for i in usd_array):
# =============================================================================
def convert_currency(instrument):
    df = candles(instrument)
    #df.index = pd.to_datetime(df.index) # change datetime to date, NOTE if granularity is less than day please delete
    df = df.reset_index()
    df.columns= ['Date','Open','High','Low','Close','Volume']
    #df.Date = pd.to_datetime(df.Date,format='%d.%m.%Y %H:%M:%S.%f')
    #df['Date'] = df['Date'].dt.date
    df = df.set_index(df.Date)
    df = df[['Open','High','Low','Close','Volume']]
    df = df.drop_duplicates(keep=False)
    
    _usd = []
    _usd= pd.DataFrame(_usd)
    
    _usd['Open'] = round(1/df['Open'],5)
    _usd['High'] = round(1/df['High'],5)   
    _usd['Low'] = round(1/df['Low'],5)
    _usd['Close'] = round(1/df['Close'],5)
    _usd['Volume'] = df['Volume']
    return _usd


#SYMBOLS = ['AUD_USD','GBP_USD','CAD_USD','CHF_USD','EUR_USD','JPY_USD','NZD_USD']
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



#usdcad = clean_format('USD_CAD')
audcad = clean_format('AUD_CAD')
#eurcad = clean_format('EUR_CAD')
nzdcad = clean_format('NZD_CAD')
#spxusd = clean_format('SPX500_USD')
#au200aud = clean_format('AU200_AUD')

#symbols_data = {'USD_CAD' : usdcad, 'AUD_CAD': audcad, 'NZD_CAD':nzdcad,'SPX500_USD': spxusd, 'AU200_AUD': au200aud }  #'EUR_CAD': eurcad, is not corrolated
symbols_data = {'AUD_CAD': audcad, 'NZD_CAD':nzdcad}

TRADING_INSTRUMENT = var_prod_1.TRADING_INSTRUMENT
SYMBOLS = ['NZD_CAD','AUD_CAD']
#SYMBOLS = ['USD_CAD','AUD_CAD','NZD_CAD','AU200_AUD','SPX500_USD'] #,'SPX500_USD'
#symbols_data = {  'JPY_USD': jpyusd, 'CHF_USD': chfusd, 'AUD_USD' : audusd, 'NZD_USD': nzdusd, 'EUR_USD': eurusd, 'GBP_USD': gbpusd, 'CAD_USD': cadusd}
for symbol in SYMBOLS:
    data = symbols_data[symbol]

SMA_NUM_PERIODS = 20  # look back period
price_history = {}  # history of prices

PRICE_DEV_NUM_PRICES = 200 # look back period of ClosePrice deviations from SMA
price_deviation_from_sma = {}  # history of ClosePrice deviations from SMA

# We will use this to iterate over all the days of data we have
num_days = len(symbols_data[TRADING_INSTRUMENT].index)
correlation_history = {} # history of correlations per currency pair
delta_projected_actual_history = {} # history of differences between Projected ClosePrice deviation and actual ClosePrice deviation per currency pair

final_delta_projected_history = [] # history of differences between final Projected ClosePrice deviation for TRADING_INSTRUMENT and actual ClosePrice deviation

# Variables for Trading Strategy trade, position & pnl management:
orders = []  # Container for tracking buy/sell order, +1 for buy order, -1 for sell order, 0 for no-action
positions = []  # Container for tracking positions, +ve for long positions, -ve for short positions, 0 for flat/no position
pnls = []  # Container for tracking total_pnls, this is the sum of closed_pnl i.e. pnls already locked in and open_pnl i.e. pnls for open-position marked to market price

last_buy_price = 0  # Price at which last buy trade was made, used to prevent over-trading at/around the same price
last_sell_price = 0  # Price at which last sell trade was made, used to prevent over-trading at/around the same price
position = 0  # Current position of the trading strategy
buy_sum_price_qty = 0  # Summation of products of buy_trade_price and buy_trade_qty for every buy Trade made since last time being flat
buy_sum_qty = 0  # Summation of buy_trade_qty for every buy Trade made since last time being flat
sell_sum_price_qty = 0  # Summation of products of sell_trade_price and sell_trade_qty for every sell Trade made since last time being flat
sell_sum_qty = 0  # Summation of sell_trade_qty for every sell Trade made since last time being flat
open_pnl = 0  # Open/Unrealized PnL marked to market
closed_pnl = 0  # Closed/Realized PnL so far

# Constants that define strategy behavior/thresholds
StatArb_VALUE_FOR_BUY_ENTRY = var_prod_1.VALUE_FOR_BUY_ENTRY  # StatArb trading signal value aboe which to enter buy-orders/long-position
StatArb_VALUE_FOR_SELL_ENTRY = var_prod_1.VALUE_FOR_SELL_ENTRY  # StatArb trading signal value below which to enter sell-orders/short-position
MIN_PRICE_MOVE_FROM_LAST_TRADE = 0  # Minimum price change since last trade before considering trading again, this is to prevent over-trading at/around same prices
NUM_SHARES_PER_TRADE = var_prod_1.NUM_SHARES_PER_TRADE  # Number of currency to buy/sell on every trade
MIN_PROFIT_TO_CLOSE = var_prod_1.pnl_value  # Minimum Open/Unrealized profit at which to close positions and lock profits

# =============================================================================
# Quantifying and computing StatArb trading signals
# =============================================================================

#i = 0
# (symbols_data[symbol].index)[num_days-1]
#i = num_days-1
for i in range(0, num_days):
  close_prices = {}

  # 1
  # Build ClosePrice series, compute SMA for each symbol and price-deviation from SMA for each symbol
  for symbol in SYMBOLS:
    close_prices[symbol] = symbols_data[symbol]['Close'].iloc[i]
    if not symbol in price_history.keys():
      price_history[symbol] = []
      price_deviation_from_sma[symbol] = []

    price_history[symbol].append(close_prices[symbol])
    if len(price_history[symbol]) > SMA_NUM_PERIODS:  # we track at most SMA_NUM_PERIODS number of prices
      del (price_history[symbol][0])

    sma = stats.mean(price_history[symbol]) # Rolling SimpleMovingAverage
    price_deviation_from_sma[symbol].append(close_prices[symbol] - sma) # price deviation from mean
    if len(price_deviation_from_sma[symbol]) > PRICE_DEV_NUM_PRICES:
      del (price_deviation_from_sma[symbol][0])
      
  # 2
  """
  Next, we need to compute the relationships between the CAD/USD currency pair price deviations and the other currency pair price deviations. 
  We will use covariance and correlation between the series of price deviations from SMA that we computed in the previous section. In this same loop, 
  we will also compute the CAD/USD price deviation as projected by every other lead currency pair, and see what the difference between 
  the projected price deviation and actual price deviation is. We will need these individual deltas between projected price deviation and 
  actual price deviation to get a final delta value that we will use for trading.
  """
  # Now compute covariance and correlation between TRADING_INSTRUMENT and every other lead symbol
  # also compute projected price deviation and find delta between projected and actual price deviations.
  projected_dev_from_sma_using = {}
  for symbol in SYMBOLS:
    if symbol == TRADING_INSTRUMENT:  # no need to find relationship between trading instrument and itself
      continue

    correlation_label = TRADING_INSTRUMENT + '<-' + symbol
    if correlation_label not in correlation_history.keys(): # first entry for this pair in the history dictionary
      correlation_history[correlation_label] = []
      delta_projected_actual_history[correlation_label] = []

    if len(price_deviation_from_sma[symbol]) < 2: # need atleast two observations to compute covariance/correlation
      correlation_history[correlation_label].append(0)
      delta_projected_actual_history[correlation_label].append(0)
      continue
    """
    Now, let's look at the code block to compute correlation and covariance between the currency pairs:
    """
    corr = np.corrcoef(price_deviation_from_sma[TRADING_INSTRUMENT], price_deviation_from_sma[symbol])
    cov = np.cov(price_deviation_from_sma[TRADING_INSTRUMENT], price_deviation_from_sma[symbol])
    corr_trading_instrument_lead_instrument = corr[0, 1]  # get the correlation between the 2 series
    cov_trading_instrument_lead_instrument = cov[0, 0] / cov[0, 1] # get the covariance between the 2 series

    correlation_history[correlation_label].append(corr_trading_instrument_lead_instrument)

    """
    Finally, let's look at the code block that computes the projected price movement, uses that to find the difference between 
    the projected movement and actual movement, and saves it in our delta_projected_actual_history list per currency pair:
    """
    # projected-price-deviation-in-TRADING_INSTRUMENT is covariance * price-deviation-in-lead-symbol
    projected_dev_from_sma_using[symbol] = price_deviation_from_sma[symbol][-1] * cov_trading_instrument_lead_instrument

    # delta +ve => signal says TRADING_INSTRUMENT price should have moved up more than what it did
    # delta -ve => signal says TRADING_INSTRUMENT price should have moved down more than what it did.
    delta_projected_actual = (projected_dev_from_sma_using[symbol] - price_deviation_from_sma[TRADING_INSTRUMENT][-1])
    delta_projected_actual_history[correlation_label].append(delta_projected_actual)
  #3
  """
  Let's combine these individual deltas between projected and actual price deviation in CAD/USD to get one final StatArb signal value for CAD/USD 
  that is a combination of projections from all the other currency pairs. To combine these different projections, we will use the magnitude of the correlation 
  between CAD/USD and the other currency pairs to weigh the delta between projected and actual price deviations in CAD/USD as predicted by the other pairs. 
  Finally, we will normalize the final delta value by the sum of each individual weight (magnitude of correlation) and 
  that is what we will use as our final signal to build our trading strategy around:
  """
  # weigh predictions from each pair, weight is the correlation between those pairs
  sum_weights = 0 # sum of weights is sum of correlations for each symbol with TRADING_INSTRUMENT
  for symbol in SYMBOLS:
    if symbol == TRADING_INSTRUMENT:  # no need to find relationship between trading instrument and itself
      continue

    correlation_label = TRADING_INSTRUMENT + '<-' + symbol
    sum_weights += abs(correlation_history[correlation_label][-1])

  final_delta_projected = 0 # will hold final prediction of price deviation in TRADING_INSTRUMENT, weighing projections from all other symbols.
  close_price = close_prices[TRADING_INSTRUMENT]
  for symbol in SYMBOLS:
    if symbol == TRADING_INSTRUMENT:  # no need to find relationship between trading instrument and itself
      continue

    correlation_label = TRADING_INSTRUMENT + '<-' + symbol

    # weight projection from a symbol by correlation
    final_delta_projected += (abs(correlation_history[correlation_label][-1]) * delta_projected_actual_history[correlation_label][-1])

  # normalize by diving by sum of weights for all pairs
  if sum_weights != 0:
    final_delta_projected /= sum_weights
  else:
    final_delta_projected = 0

  final_delta_projected_history.append(final_delta_projected)


#output variable 
output_value = final_delta_projected
with open("C:\\Users\\gutia\\Anaconda3\\ForestTrade\\log\\prod_1_a2_final_delta_projected.txt","a+") as f:
    f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) +'  value: '+str(output_value)+ '\n')
    f.close()

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) +'  value: '+str(output_value))