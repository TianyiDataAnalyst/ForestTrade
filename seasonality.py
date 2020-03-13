#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 22:52:21 2020

@author: tianyigu
"""

df['DateTime'] = pd.to_datetime(df['DateTime'], format="%Y%m%d%H%M%S")
df = df.reset_index(drop = True)
df = df.set_index('DateTime')
df = df.drop(['Date','Time'],axis = 1)


#Adding data to help the model deal with seasonality

#df['DayOfYear'] = df.index.dayofyear
df['HourOfDay'] = df.index.hour
#df['MonthOfYear'] = df.index.month

