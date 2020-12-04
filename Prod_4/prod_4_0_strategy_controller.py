# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:14:14 2020

@author: gutia
"""
import time
import os
#from ForestTrade.file_directory import file_name

def main():
    os.system('python C:\\Users\\gutia\\Anaconda3\\ForestTrade\\Prod_4\\prod_4_5_StatArbitrage_strategy.py')
# run 12 hours and trigger the file in every 15 minutes
starttime=time.time()
timeout = time.time() + (60*60*24*16)  # 60 seconds times 60 meaning the script will run for 1 hr
while time.time() <= timeout:
    try:
        print("prod_4_Strategy_controler script passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main()
        time.sleep(15*60 - ((time.time() - starttime) % 15.0*60)) # orignial 300=5 minute interval between each new execution
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()