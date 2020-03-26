# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:14:14 2020

@author: gutia
"""
import time
import os
 
def main():
    os.system('python C:\\Users\\gutia\\Documents\\GitHub\\ForestTrade\\StatAbit_strategy.py')

starttime=time.time()
timeout = time.time() + (60*60*12)  # 60 seconds times 60 meaning the script will run for 1 hr
while time.time() <= timeout:
    try:
        print("passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main()
        time.sleep(15*60 - ((time.time() - starttime) % 15.0*60)) # orignial 300=5 minute interval between each new execution
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()