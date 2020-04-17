# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 23:03:31 2020

@author: gutia
"""

import oandapyV20
#import matplotlib.pyplot as plt
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


import oandapyV20.endpoints.positions as positions
def position_close():
    accountID = account_id
    instrument = var_prod_1.TRADING_INSTRUMENT
    data ={
           "longUnits": "ALL"
           }
    r = positions.PositionClose(accountID=accountID,instrument=instrument,data=data)
    client.request(r)
    return r.response
