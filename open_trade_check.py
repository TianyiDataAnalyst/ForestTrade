# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:20:54 2020

@author: gutia
"""

import oandapyV20
import oandapyV20.endpoints.trades as trades
import oandapyV20.definitions.instruments as definstruments
from ForestTrade.config import token
from ForestTrade.config import oanda_login as account
from ForestTrade.config import var_prod_1


# =============================================================================
# pd.set_option('display.max_rows', 1000) 
# pd.set_option('display.width', None)   
# pd.set_option('display.max_colwidth', 1000)
# =============================================================================

#initiating API connection and defining trade parameters
CandlestickGranularity = (definstruments.CandlestickGranularity().definitions.keys())

#initiating API connection and defining trade parameters
client = oandapyV20.API(token.token,environment="practice")
account_id = account.oanda_pratice_account_id

#Globle variable 
trade_instrument = var_prod_1.TRADING_INSTRUMENT
NUM_SHARES_PER_TRADE = var_prod_1.NUM_SHARES_PER_TRADE 

#defining strategy parameters
#pairs = ['AUD_USD','GBP_USD','USD_CAD','USD_CHF','EUR_USD','USD_JPY','NZD_USD'] #currency pairs to be included in the strategy
#pairs = ['EUR_JPY','USD_JPY','AUD_JPY','AUD_USD','AUD_NZD','NZD_USD']

#instrument = 'NZD_CAD'
r = trades.OpenTrades(accountID=account_id)
open_trades = client.request(r)['trades']
range(len(open_trades))