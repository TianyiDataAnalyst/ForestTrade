# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 08:26:41 2020

@author: gutia
"""

import oandapyV20
import oandapyV20.endpoints.orders as orders

#

#initiating API connection and defining trade parameters
token_path = "C:\\Oanda\\token.txt"
client = oandapyV20.API(access_token=open(token_path,'r').read(),environment="practice")
account_id = "101-002-9736246-001"

#orders
data = {
        "order": {
        "price": "1.15",
        "stopLossOnFill": {
        "timeInForce": "GTC",
        "price": "1.2"
                          },
        "timeInForce": "FOK",
        "instrument": "USD_CAD",
        "units": "100",
        "type": "MARKET",
        "positionFill": "DEFAULT"
                }
        }
            
r = orders.OrderCreate(accountID=account_id, data=data)
client.request(r)
