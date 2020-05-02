# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 02:09:45 2020

@author: gutia
"""
import oandapyV20
import oandapyV20.endpoints.trades as trades

def trade_close_single(trade_ID): #OANDA REST-V20 API Documentation,Page 6
        accountID= account.oanda_pratice_account_id
        tradeID = trade_ID
        cfg ={"units": str(var_prod_1.NUM_SHARES_PER_TRADE)}
        r = trades.TradeClose(accountID, tradeID, data=cfg)
        client.request(r)

#oandapyV20.API.request(trade_close_single(trade_ID))
#trade_close('6411','2000')
#trade_ID ='2876'
def main():
    r = trades.OpenTrades(accountID=account_id)
    open_trades = client.request(r)['trades']
    trade_ids = []
    trade_id = []
    risk_distance = var_prod_1.risk_distance
    if len(open_trades)==0:
        print("no open trade: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))   
    for i in range(len(open_trades)):
        trade_ids.append(open_trades[i]['id'])
    trade_id = [i for i in trade_ids if i not in trade_ids]
    for trade_id in trade_ids:       
        if  current_unit(trade_id) > 0 and BollBnd(candles_h1(TRADING_INSTRUMENT),20)['BB_width'][-1] >float(risk_distance) and candles_h1(TRADING_INSTRUMENT)['c'][-1] > BollBnd(candles_h1(TRADING_INSTRUMENT),20)['BB_up'][-1]: 
            print("Close long trade:")
            print("ID:", trade_id)
            trade_close_single(trade_id)
            f = open(file_name('log\\prod_1_pnl.txt'),"a+")
            f.write(oandapyV20.API.request(trade_close_single(trade_id)))
            f.close
        if current_unit(trade_id) < 0 and BollBnd(candles_h1(TRADING_INSTRUMENT),20)['BB_width'][-1] >float(risk_distance) and candles_h1(TRADING_INSTRUMENT)['c'][-1] < BollBnd(candles_h1(TRADING_INSTRUMENT),20)['BB_dn'][-1]:
            print("Close short trade:")
            print("ID:", trade_id)
            trade_close_single(trade_id)
            f = open(file_name('log\\prod_1_pnl.txt'),"a+")
            f.write(oandapyV20.API.request(trade_close_single(trade_id)))
            f.close