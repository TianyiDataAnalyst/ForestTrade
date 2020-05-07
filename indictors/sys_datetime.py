# -*- coding: utf-8 -*-
"""
Created on Mon May  4 11:56:18 2020

@author: gutia
"""

import oandapyV20
import oandapyV20.endpoints.trades as trades
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import time
import oandapyV20.endpoints.forexlabs as labs
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.definitions.instruments as definstruments
import oandapyV20.endpoints.pricing as pricing
import datetime as dt
import re
from ForestTrade.config import token
from ForestTrade.config import oanda_login_prod_2 as account
from ForestTrade.config import var_prod_2
from ForestTrade.file_directory import file_name

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


currency ='USD_CAD'

def sys_time(currency):
    params = {"instruments": currency}
    account_id = account.oanda_pratice_account_id
    r = pricing.PricingInfo(accountID=account_id, params=params)
    rv = client.request(r)
    print(rv["time"])