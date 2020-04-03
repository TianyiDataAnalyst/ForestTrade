#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:27:05 2020

@author: tianyigu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:36:39 2020

@author: gutia
"""

# =============================================================================
# import pandas as pd
# from pandas_datareader import data
# =============================================================================

# Fetch daily data for 4 years
# =============================================================================
# SYMBOL='GOOG'
# start_date = '2014-01-01'
# end_date = '2018-01-01'
# SRC_DATA_FILENAME=SYMBOL + '_data.pkl'
# 
# try:
#  data = pd.read_pickle(SRC_DATA_FILENAME)
# except FileNotFoundError:
#  data = data.DataReader(SYMBOL, 'yahoo', start_date, end_date)
#  data.to_pickle(SRC_DATA_FILENAME)
# 
# =============================================================================
#data.head(4)
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.definitions.instruments as definstruments
import pandas as pd
#import matplotlib.pyplot as plt
import statistics as stats
import numpy as np

from config import oanda_login as account

#
CandlestickGranularity = (definstruments.CandlestickGranularity().definitions.keys())

#initiating API connection and defining trade parameters
token_path = "C:\\Oanda\\token.txt" # Windows system format: "C:\\Oanda\\token.txt"; "token.txt" in PyCharm; ios "/Users/tianyigu/Downloads/token.txt"
client = oandapyV20.API(access_token=open(token_path,'r').read(),environment="practice")
account_id = account.oanda_pratice_account_id



#defining strategy parameters
#pairs = ['AUD_USD','GBP_USD','USD_CAD','USD_CHF','EUR_USD','USD_JPY','NZD_USD'] #currency pairs to be included in the strategy
#pairs = ['EUR_JPY','USD_JPY','AUD_JPY','AUD_USD','AUD_NZD','NZD_USD']

def candles(instrument):
    params = {"count": 1500,"granularity": list(CandlestickGranularity)[9]} #granularity is in 'M15'; it can be in seconds S5 - S30, minutes M1 - M30, hours H1 - H12, days D[18], weeks W or months M
    candles = instruments.InstrumentsCandles(instrument=instrument,params=params)
    client.request(candles)
    ohlc_dict = candles.response["candles"]
    ohlc = pd.DataFrame(ohlc_dict)
    ohlc_df = ohlc.mid.dropna().apply(pd.Series)
    ohlc_df["volume"] = ohlc["volume"]
    ohlc_df.index = ohlc["time"]
    ohlc_df = ohlc_df.apply(pd.to_numeric)
    return ohlc_df
#candles('AUD_CAD')

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


#USD based data
#dictionary structure:https://realpython.com/python-dicts/
# =============================================================================
# jpyusd = convert_currency('USD_JPY')
# #jpyusd['Open'].count()
# #jpyusd = jpyusd.assign(symbol=pd.Series('JPY_USD', index=jpyusd.index))
# #jpyusd_dict = jpyusd.groupby('symbol').apply(lambda dfg: dfg.drop('symbol', axis=1).to_dict(orient='list')).to_dict()
# 
# chfusd = convert_currency('USD_CHF')
# 
# #chfusd = chfusd.assign(symbol=pd.Series('CHF_USD', index=chfusd.index))
# #chfusd_dict = chfusd.groupby('symbol').apply(lambda dfg: dfg.drop('symbol', axis=1).to_dict(orient='list')).to_dict()
# 
# audusd = clean_format('AUD_USD')
# #audusd['Open'].count()
# #audusd = audusd.assign(symbol=pd.Series('AUD_USD', index=audusd.index))
# #audusd_dict = audusd.groupby('symbol').apply(lambda dfg: dfg.drop('symbol', axis=1).to_dict(orient='list')).to_dict()
# 
# #['GBP_USD','CAD_USD','EUR_USD','NZD_USD']
# cadusd = convert_currency('USD_CAD')
# 
# gbpusd = clean_format('GBP_USD')
# eurusd = clean_format('EUR_USD')
# nzdusd = clean_format('NZD_USD')
# spxusd = clean_format('SPX500_USD')
# =============================================================================

usdcad = clean_format('USD_CAD')
audcad = clean_format('AUD_CAD')
#eurcad = clean_format('EUR_CAD')
nzdcad = clean_format('NZD_CAD')
spxusd = clean_format('SPX500_USD')
au200aud = clean_format('AU200_AUD')

#symbols_data = {'USD_CAD' : usdcad, 'AUD_CAD': audcad, 'NZD_CAD':nzdcad,'SPX500_USD': spxusd, 'AU200_AUD': au200aud }  #'EUR_CAD': eurcad, is not corrolated
symbols_data = {'AUD_CAD': audcad, 'NZD_CAD':nzdcad}

TRADING_INSTRUMENT = 'NZD_CAD'
#SYMBOLS = ['USD_CAD','AUD_CAD','NZD_CAD','AU200_AUD','SPX500_USD'] #,'SPX500_USD'
SYMBOLS = ['NZD_CAD','AUD_CAD']
#symbols_data = {'AUD_USD' : audusd, 'NZD_USD': nzdusd, 'CAD_USD': cadusd,'SPX500_USD': spxusd}
#symbols_data = {  'JPY_USD': jpyusd, 'CHF_USD': chfusd, 'AUD_USD' : audusd, 'NZD_USD': nzdusd, 'EUR_USD': eurusd, 'GBP_USD': gbpusd, 'CAD_USD': cadusd}
for symbol in SYMBOLS:
    data = symbols_data[symbol]

# Visualize prices for currency to inspect relationship between them
import matplotlib.pyplot as plt

from itertools import cycle

cycol = cycle('bgrcmky')

price_data = pd.DataFrame()
for symbol in SYMBOLS:
  multiplier = 1.0
  if symbol == 'SPX500_USD':
    multiplier = 1/2800
  if symbol == 'AU200_AUD':
      multiplier = 1/9000
      
  label = symbol + ' ClosePrice'
  price_data = price_data.assign(label=pd.Series(symbols_data[symbol]['Close'] * multiplier, index=symbols_data[symbol].index))
  ax = price_data['label'].plot(color=next(cycol), lw=2., label=label)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Scaled Price', fontsize=18)
plt.legend(prop={'size': 18})
plt.show()
# =============================================================================
# data = {jpyusd, chfusd, audusd, gbpusd}
# 
# gbpusd = clean_format('GBP_USD')
# gbpusd = gbpusd.assign(symbol=pd.Series('GBP_USD', index=gbpusd.index))
# gbpusd_dict = gbpusd.groupby('symbol').apply(lambda dfg: dfg.drop('symbol', axis=1).to_dict(orient='list')).to_dict()
# 
# 
# jpsnchf = pd.concat([jpyusd, chfusd, audusd, gbpusd])
# 
# all_dict = jpsnchf.groupby('symbol').apply(lambda dfg: dfg.drop('symbol', axis=1).to_dict(orient='list')).to_dict()
# =============================================================================

# Constants/variables that are used to compute simple moving average and price deviation from simple moving average
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
StatArb_VALUE_FOR_BUY_ENTRY = 0.001  # StatArb trading signal value aboe which to enter buy-orders/long-position
StatArb_VALUE_FOR_SELL_ENTRY = -0.001  # StatArb trading signal value below which to enter sell-orders/short-position
MIN_PRICE_MOVE_FROM_LAST_TRADE = 0.001  # Minimum price change since last trade before considering trading again, this is to prevent over-trading at/around same prices
NUM_SHARES_PER_TRADE = 2000  # Number of currency to buy/sell on every trade
MIN_PROFIT_TO_CLOSE = 10  # Minimum Open/Unrealized profit at which to close positions and lock profits


# =============================================================================
# Quantifying and computing StatArb trading signals
# =============================================================================

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

# =============================================================================
# StatArb execution logic
# =============================================================================

  # 1
  # This section checks trading signal against trading parameters/thresholds and positions, to trade.
  #
  # We will perform a sell trade at close_prices if the following conditions are met:
  # 1. The StatArb trading signal value is below Sell-Entry threshold and the difference between last trade-price and current-price is different enough.
  # 2. We are long( +ve position ) and current position is profitable enough to lock profit.
  if ((final_delta_projected < StatArb_VALUE_FOR_SELL_ENTRY and abs(close_price - last_sell_price) > MIN_PRICE_MOVE_FROM_LAST_TRADE)  # StatArb above sell entry threshold, we should sell
      or
      (position > 0 and (open_pnl > MIN_PROFIT_TO_CLOSE))):  # long from -ve StatArb and StatArb has gone positive or position is profitable, sell to close position
    orders.append(-1)  # mark the sell trade
    last_sell_price = close_price
    position -= NUM_SHARES_PER_TRADE  # reduce position by the size of this trade
    sell_sum_price_qty += (close_price * NUM_SHARES_PER_TRADE)  # update vwap sell-price
    sell_sum_qty += NUM_SHARES_PER_TRADE
    #simulate trade
    #print trade result 
# =============================================================================
#     print("Sell ", NUM_SHARES_PER_TRADE, " @ ", close_price, "Position: ", position)
#     print("OpenPnL: ", open_pnl, " ClosedPnL: ", closed_pnl, " TotalPnL: ", (open_pnl + closed_pnl))
# =============================================================================

  # 2
  # We will perform a buy trade at close_prices if the following conditions are met:
  # 1. The StatArb trading signal value is above Buy-Entry threshold and the difference between last trade-price and current-price is different enough.
  # 2. We are short( -ve position ) and current position is profitable enough to lock profit.
  elif ((final_delta_projected > StatArb_VALUE_FOR_BUY_ENTRY and abs(close_price - last_buy_price) > MIN_PRICE_MOVE_FROM_LAST_TRADE)  # StatArb below buy entry threshold, we should buy
        or
        (position < 0 and (open_pnl > MIN_PROFIT_TO_CLOSE))):  # short from +ve StatArb and StatArb has gone negative or position is profitable, buy to close position
    orders.append(+1)  # mark the buy trade
    last_buy_price = close_price
    position += NUM_SHARES_PER_TRADE  # increase position by the size of this trade
    buy_sum_price_qty += (close_price * NUM_SHARES_PER_TRADE)  # update the vwap buy-price
    buy_sum_qty += NUM_SHARES_PER_TRADE
    #simulate trade
    #print trade result    
# =============================================================================
#     print("Buy ", NUM_SHARES_PER_TRADE, " @ ", close_price, "Position: ", position)
#     print("OpenPnL: ", open_pnl, " ClosedPnL: ", closed_pnl, " TotalPnL: ", (open_pnl + closed_pnl))
# =============================================================================
  else:
    # No trade since none of the conditions were met to buy or sell
    orders.append(0)

  positions.append(position)

  #3
  """
  Finally, let's also look at the position management and PnL update logic, very similar to previous trading strategies:
  """
  # This section updates Open/Unrealized & Closed/Realized positions
  open_pnl = 0
  if position > 0:
    if sell_sum_qty > 0:  # long position and some sell trades have been made against it, close that amount based on how much was sold against this long position
      open_pnl = abs(sell_sum_qty) * (sell_sum_price_qty / sell_sum_qty - buy_sum_price_qty / buy_sum_qty)
    # mark the remaining position to market i.e. pnl would be what it would be if we closed at current price
    open_pnl += abs(sell_sum_qty - position) * (close_price - buy_sum_price_qty / buy_sum_qty)
  elif position < 0:
    if buy_sum_qty > 0:  # short position and some buy trades have been made against it, close that amount based on how much was bought against this short position
      open_pnl = abs(buy_sum_qty) * (sell_sum_price_qty / sell_sum_qty - buy_sum_price_qty / buy_sum_qty)
    # mark the remaining position to market i.e. pnl would be what it would be if we closed at current price
    open_pnl += abs(buy_sum_qty - position) * (sell_sum_price_qty / sell_sum_qty - close_price)
  else:
    # flat, so update closed_pnl and reset tracking variables for positions & pnls
    closed_pnl += (sell_sum_price_qty - buy_sum_price_qty)
    buy_sum_price_qty = 0
    buy_sum_qty = 0
    sell_sum_price_qty = 0
    sell_sum_qty = 0
    last_buy_price = 0
    last_sell_price = 0

  pnls.append(closed_pnl + open_pnl)




# =============================================================================
# StatArb signal and strategy performance analysis
# =============================================================================
  
  
#plot 2
"""
Let's visualize a few more details about the signals in this trading strategy, starting with the correlations 
between CAD/USD and the other currency pairs as it evolves over time:

"""
# Plot correlations between TRADING_INSTRUMENT and other currency pairs

correlation_data = pd.DataFrame()
for symbol in SYMBOLS:
  if symbol == TRADING_INSTRUMENT:
    continue

  correlation_label = TRADING_INSTRUMENT + '<-' + symbol
  correlation_data = correlation_data.assign(label=pd.Series(correlation_history[correlation_label], index=symbols_data[symbol].index))
  ax = correlation_data['label'].plot(color=next(cycol), lw=2., label='Correlation ' + correlation_label)

for i in np.arange(-1, 1, 0.25):
  plt.axhline(y=i, lw=0.5, color='k')
plt.legend()
plt.show()

"""
This plot shows the correlation between CADUSD and other currency pairs as it evolves over the course of this trading strategy. 
Correlations close to -1 or +1 signify strongly correlated pairs, and correlations that hold steady are the stable correlated pairs. 
Currency pairs where correlations swing around between negative and positive values indicate extremely uncorrelated or unstable currency pairs, 
which are unlikely to yield good predictions in the long run. However, we do not know how the correlation would evolve ahead of time, 
so we have no choice but to use all currency pairs available to us in our StatArb trading strategy:
As we suspected, the currency pairs that are most strongly correlated to CAD/USD price deviations are AUD/USD and NZD/USD. 
JPY/USD is the least correlated to CAD/USD price deviations.
"""


#plot 3
"""
Now, let's inspect the delta between projected and actual price deviations in CAD/USD as projected by 
each individual currency pair individually:
"""
# Plot StatArb signal provided by each currency pair
delta_projected_actual_data = pd.DataFrame()
for symbol in SYMBOLS:
  if symbol == TRADING_INSTRUMENT:
    continue

  projection_label = TRADING_INSTRUMENT + '<-' + symbol
  delta_projected_actual_data = delta_projected_actual_data.assign(StatArbTradingSignal=pd.Series(delta_projected_actual_history[projection_label], index=symbols_data[TRADING_INSTRUMENT].index))
  ax = delta_projected_actual_data['StatArbTradingSignal'].plot(color=next(cycol), lw=1., label='StatArbTradingSignal ' + projection_label)
plt.legend()
plt.show()
"""
This is what the StatArb signal values would look like if we used any of the currency pairs alone to project CAD/USD price deviations:
   ~ Chart ~
Here, the plot seems to suggest that JPYUSD and CHFUSD have very large predictions, but as we saw before those pairs do not have good correlations with CADUSD, 
so these are likely to be bad predictions due to poor predictive relationships between CADUSD - JPYUSD and CADUSD - CHFUSD. One lesson to take away from this is that 
StatArb benefits from having multiple leading trading instruments, because when relationships break down between specific pairs, 
the other strongly correlated pairs can help offset bad predictions, which we discussed earlier.
"""

#plot 4
#Now, let's set up our data frames to plot the close price, trades, positions, and PnLs we will observe:
delta_projected_actual_data = delta_projected_actual_data.assign(ClosePrice=pd.Series(symbols_data[TRADING_INSTRUMENT]['Close'], index=symbols_data[TRADING_INSTRUMENT].index))
delta_projected_actual_data = delta_projected_actual_data.assign(FinalStatArbTradingSignal=pd.Series(final_delta_projected_history, index=symbols_data[TRADING_INSTRUMENT].index))
delta_projected_actual_data = delta_projected_actual_data.assign(Trades=pd.Series(orders, index=symbols_data[TRADING_INSTRUMENT].index))
delta_projected_actual_data = delta_projected_actual_data.assign(Position=pd.Series(positions, index=symbols_data[TRADING_INSTRUMENT].index))
delta_projected_actual_data = delta_projected_actual_data.assign(Pnl=pd.Series(pnls, index=symbols_data[TRADING_INSTRUMENT].index))

plt.plot(delta_projected_actual_data.index, delta_projected_actual_data.ClosePrice, color='k', lw=1., label='ClosePrice')
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Trades == 1].index, delta_projected_actual_data.ClosePrice[delta_projected_actual_data.Trades == 1], color='r', lw=0, marker='^', markersize=7, label='buy')
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Trades == -1].index, delta_projected_actual_data.ClosePrice[delta_projected_actual_data.Trades == -1], color='g', lw=0, marker='v', markersize=7, label='sell')
plt.legend()
plt.show()
"""
The following plot tells us at what prices the buy and sell trades are made in CADUSD. We will need to inspect the final trading signal 
in addition to this plot to fully understand the behavior of this StatArb signal and strategy:
    ~ Chart ~
Now, let's look at the actual code to build visualization for the final StatArb trading signal, and overlay buy and sell trades over the lifetime of the signal evolution. 
This will help us understand for what signal values buy and sell trades are made and if that is in line with our expectations: (connect to next plot analysis)
"""

#plot 5
"""
Since we adopted the trend-following approach in our StatArb trading strategy, we expect to buy when the signal value is positive and sell when the signal value is negative. Let's see whether that's the case in the plot:
"""
plt.plot(delta_projected_actual_data.index, delta_projected_actual_data.FinalStatArbTradingSignal, color='k', lw=1., label='FinalStatArbTradingSignal')
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Trades == 1].index, delta_projected_actual_data.FinalStatArbTradingSignal[delta_projected_actual_data.Trades == 1], color='r', lw=0, marker='^', markersize=7, label='buy')
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Trades == -1].index, delta_projected_actual_data.FinalStatArbTradingSignal[delta_projected_actual_data.Trades == -1], color='g', lw=0, marker='v', markersize=7, label='sell')
plt.axhline(y=0, lw=0.5, color='k')
for i in np.arange(StatArb_VALUE_FOR_BUY_ENTRY, StatArb_VALUE_FOR_BUY_ENTRY * 10, StatArb_VALUE_FOR_BUY_ENTRY * 2):
  plt.axhline(y=i, lw=0.5, color='r')
for i in np.arange(StatArb_VALUE_FOR_SELL_ENTRY, StatArb_VALUE_FOR_SELL_ENTRY * 10, StatArb_VALUE_FOR_SELL_ENTRY * 2):
  plt.axhline(y=i, lw=0.5, color='g')
plt.legend()
plt.show()
"""
Based on this plot and our understanding of trend-following strategies in addition to the StatArb signal we built, 
we do indeed see many buy trades when the signal value is positive and sell trades when the signal values are negative. 
The buy trades made when signal values are negative and sell trades made when signal values are positive can be attributed to the trades that close profitable positions, 
as we saw in our previous mean reversion and trend-following trading strategies.
"""

#plot 6
"""
Let's wrap up our analysis of StatArb trading strategies by visualizing the positions and PnLs:
"""
plt.plot(delta_projected_actual_data.index, delta_projected_actual_data.Position, color='k', lw=1., label='Position')
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Position == 0].index, delta_projected_actual_data.Position[delta_projected_actual_data.Position == 0], color='k', lw=0, marker='.', label='flat')
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Position > 0].index, delta_projected_actual_data.Position[delta_projected_actual_data.Position > 0], color='r', lw=0, marker='+', label='long')
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Position < 0].index, delta_projected_actual_data.Position[delta_projected_actual_data.Position < 0], color='g', lw=0, marker='_', label='short')
plt.axhline(y=0, lw=0.5, color='k')
for i in range(NUM_SHARES_PER_TRADE, NUM_SHARES_PER_TRADE * 5, NUM_SHARES_PER_TRADE):
  plt.axhline(y=i, lw=0.5, color='r')
for i in range(-NUM_SHARES_PER_TRADE, -NUM_SHARES_PER_TRADE * 5, -NUM_SHARES_PER_TRADE):
  plt.axhline(y=i, lw=0.5, color='g')
plt.legend()
plt.show()
"""
The position plot shows the evolution of the StatArb trading strategy's position over the course of its lifetime. 
Remember that these positions are in dollar notional terms, so a position of 100K is equivalent to roughly 1 future contract, 
which we mention to make it clear that a position of 100K does not mean a position of 100K contracts!
"""
#plot 7
# Finally, let's have a look at the code for the PnL plot, identical to what we've been using before:
plt.plot(delta_projected_actual_data.index, delta_projected_actual_data.Pnl, color='k', lw=1., label='Pnl')
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Pnl > 0].index, delta_projected_actual_data.Pnl[delta_projected_actual_data.Pnl > 0], color='g', lw=0, marker='.')
plt.plot(delta_projected_actual_data.loc[delta_projected_actual_data.Pnl < 0].index, delta_projected_actual_data.Pnl[delta_projected_actual_data.Pnl < 0], color='r', lw=0, marker='.')
plt.legend()
plt.show()
"""
We expect to see better performance here than in our previously built trading strategies because it relies on a fundamental relationship 
between different currency pairs and should be able to perform better during 
different market conditions because of its use of multiple currency pairs as lead trading instruments:
"""
delta_projected_actual_data.to_csv("statistical_arbitrage.csv", sep=",")

"""
And that's it, now you have a working example of a profitable statistical arbitrage strategy and should be able to improve and extend it to other trading instruments!
"""
