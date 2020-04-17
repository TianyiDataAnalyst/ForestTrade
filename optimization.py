# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:07:54 2020

@author: gutia
"""

import time
import random

PREFIX = 0
THRESH = float(0.00)
THRESH2 = float(0.00)
INTERVAL = 0

# load data in to double list,
def combination():
    prefix = []
    thresh = []
    interval = [1,2,3,5,10,15]
    combinedList = []
    for i in range(2,21):
        prefix.append(i)
    threshBasic = float(0.001)
    for i in range(1,8):
        thresh.append(float(threshBasic*i))
    #
    # print(prefix)
    # print(thresh)
    # print(interval)

    for p in prefix:
        for t in thresh:
            for i in interval:
                li = []
                li.append(p)
                li.append(t)
                li.append(-t)
                li.append(i)
                combinedList.append(li)

    return combinedList

def main1():
    result = combination()



def main():
    #before run
    global PREFIX
    global THRESH
    global THRESH2
    global INTERVAL
    datas = combination()
    result = []

    # looping combination data
    # for each loop, update global variable
    # run script and save in txt file
    # sleep 15 mins
    for ele in datas:
        PREFIX = ele[0]
        THRESH = ele[1]
        THRESH2 = ele[2]
        INTERVAL = ele[3]
        #####
        ###
        profit = random.randint(0,30)

        ###
        ###
        print(PREFIX,THRESH,THRESH2,INTERVAL)

        time.sleep(5)

main()