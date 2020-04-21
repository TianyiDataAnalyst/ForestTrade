# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:53:23 2020

@author: gutia
"""

#pip_price= 0.02 CAD
#1=S10;3=S30;9=M15;12=H2;13=H13;14=H4;15=H6;18=D;19=W
#margin available = 2000; Maximum Number Of Units = 112305; $35.62CAD per trade(2000 units)
#test result: max position= 125000; so account should be (125000/2000)*35.62>=1,296‬
NUM_SHARES_PER_TRADE = 2000
VALUE_FOR_BUY_ENTRY = 0.000829
VALUE_FOR_SELL_ENTRY =-0.000829
TRADING_INSTRUMENT = 'NZD_CAD'
#$2CAD = 10.0 pip, stop_loss = -$16.5 = 82.2pip; take profit=164 pip
pnl_value = 34
time_interval= 9
time_interval_sl = 12
risk_distance = 0.000513
