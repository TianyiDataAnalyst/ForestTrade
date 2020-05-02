# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:53:23 2020

@author: gutia
"""


TRADING_INSTRUMENT = 'NZD_CAD'

pnl_value = float(0)
NUM_SHARES_PER_TRADE = 2000

time_interval= 4
time_interval_sl = 12

risk_distance = 0.0001

rsi_buy_entry = 28
rsi_sell_entry = 72

pairs = ['AUD_USD','GBP_USD','USD_CAD','USD_CHF','EUR_USD','NZD_USD','NZD_CAD','EUR_CAD','GBP_CAD'] #currency pairs to be included in the strategy
