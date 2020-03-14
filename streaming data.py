# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 10:21:56 2020

@author: gutia
"""


import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import warnings
warnings.filterwarnings('ignore')

#initiating API connection and defining trade parameters
token_path = "C:\\Oanda\\token.txt"
client = oandapyV20.API(access_token=open(token_path,'r').read(),environment="practice")
account_id = "101-002-9736246-001"

params = {"instruments": "USD_JPY"}
account_id = account_id
r = pricing.PricingInfo(accountID=account_id, params=params)
i=0
while i <=20:
    rv = client.request(r)
    print("Time=",rv["time"])
    print("bid=",rv["prices"][0]["closeoutBid"])
    print("ask=",rv["prices"][0]["closeoutAsk"])
    print("*******************")
    i+=1
