# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:20:52 2020

@author: gutia
"""

import oandapyV20.endpoints.positions as positions
import oandapyV20

#initiating API connection and defining trade parameters
token_path = "C:\\Oanda\\token.txt"
client = oandapyV20.API(access_token=open(token_path,'r').read(),environment="practice")
account_id = "101-002-9736246-001"


r = positions.PositionDetails(accountID=account_id, instrument="USD_CAD")
client.request(r)
print (r.response)