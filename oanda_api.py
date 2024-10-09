from flask import Flask, jsonify, request
import oandapyV20
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.definitions.instruments as definstruments
import oandapyV20.endpoints.positions as positions
import pandas as pd
import numpy as np
import time
import re
import datetime as dt
from ForestTrade.config import token, oanda_login as account, var_prod_1

app = Flask(__name__)

# OANDA API Setup
client = oandapyV20.API(token.token, environment="practice")
account_id = account.oanda_pratice_account_id

# Global variables
TRADING_INSTRUMENT = var_prod_1.TRADING_INSTRUMENT
NUM_SHARES_PER_TRADE = var_prod_1.NUM_SHARES_PER_TRADE

# Utility function to format datetime
def format_datetime(date):
    conformed_timestamp = re.sub(r"[:]|([-](?!((\d{2}[:]\d{2})|(\d{4}))$))", '', date)
    x_day = dt.datetime(int(conformed_timestamp[0:4]), int(conformed_timestamp[4:6]), int(conformed_timestamp[6:8]),
                        int(conformed_timestamp[9:11]), int(conformed_timestamp[11:13]))
    return x_day

# API to get candlestick data
@app.route('/candles', methods=['GET'])
def get_candles():
    instrument = request.args.get('instrument', TRADING_INSTRUMENT)
    granularity = request.args.get('granularity', 'M15')
    params = {"count": 50, "granularity": granularity}
    candles = instruments.InstrumentsCandles(instrument=instrument, params=params)
    client.request(candles)
    ohlc_dict = candles.response["candles"]
    ohlc = pd.DataFrame(ohlc_dict).mid.dropna().apply(pd.Series)
    return jsonify(ohlc.to_dict())

# API to calculate ATR
@app.route('/atr', methods=['GET'])
def get_atr():
    instrument = request.args.get('instrument', TRADING_INSTRUMENT)
    n = int(request.args.get('n', 14))
    ohlc_df = pd.DataFrame(get_candles(instrument))
    atr = ATR(ohlc_df, n)
    return jsonify({'ATR': atr})

# ATR Calculation function
def ATR(DF, n):
    df = DF.copy()
    df['H-L'] = abs(df['h'] - df['l'])
    df['H-PC'] = abs(df['h'] - df['c'].shift(1))
    df['L-PC'] = abs(df['l'] - df['c'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=True)
    df['ATR'] = df['TR'].rolling(n).mean()
    return round(3 * df['ATR'][-1], 5)

# API to close all long positions
@app.route('/close_long', methods=['POST'])
def close_long():
    data = {"longUnits": "ALL"}
    r = positions.PositionClose(accountID=account_id, instrument=TRADING_INSTRUMENT, data=data)
    response = client.request(r)
    return jsonify(response)

# API to close all short positions
@app.route('/close_short', methods=['POST'])
def close_short():
    data = {"shortUnits": "ALL"}
    r = positions.PositionClose(accountID=account_id, instrument=TRADING_INSTRUMENT, data=data)
    response = client.request(r)
    return jsonify(response)

# API to check open trades
@app.route('/open_trades', methods=['GET'])
def open_trades():
    r = trades.OpenTrades(accountID=account_id)
    response = client.request(r)['trades']
    return jsonify(response)

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)



"""
API Endpoints:
Get Candles:

Endpoint: GET /candles
Query Parameters:
instrument (optional) – Currency pair (e.g., 'EUR_USD')
granularity (optional) – Timeframe (default: 'M15')
Get ATR:

Endpoint: GET /atr
Query Parameters:
instrument (optional) – Currency pair
n (optional) – Period for ATR calculation
Close Long Positions:

Endpoint: POST /close_long
Action: Closes all long positions on the current instrument.
Close Short Positions:

Endpoint: POST /close_short
Action: Closes all short positions on the current instrument.
Check Open Trades:

Endpoint: GET /open_trades
Action: Retrieves all open trades.

"""
                     
