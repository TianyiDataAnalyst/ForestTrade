# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:59:01 2020

@author: gutia
"""

#show final detal projected number as Scientific notation or called Standard form
def as_num(x):
    y = '{:.10f}'.format(x)  # .10f 保留10位小数
    return y

if __name__ == '__main__':
    str = '-4.90457658824787e-06' #-0.0000049046
    if ('E' in str or 'e' in str):
        x = as_num(float(str))
        print(x)