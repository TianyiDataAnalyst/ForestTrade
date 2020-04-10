# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:30:10 2020

@author: gutia
"""

import os


def readFile(filename):
    filehandle = open(filename)
    print (filehandle.read())
    filehandle.close()


fileDir = os.path.dirname(os.path.realpath('__file__'))


def file_name(filename):
    fileName = os.path.join(fileDir, 'Anaconda3\\ForestTrade\\',filename)
    return fileName
#file_name('log\\trade_log.txt')
# =============================================================================
# #For accessing the file in the same folder
# filename = "Anaconda3\\ForestTrade\\log\\trade_log.txt"
# readFile(filename)
# 
# #For accessing the file in a folder contained in the current folder
# filename = os.path.join(fileDir, 'Anaconda3\\ForestTrade\\','log\\trade_log.txt')
# readFile(filename)
# 
# 
# #For accessing the file inside a sibling folder.
# filename = os.path.join(fileDir, 'Anaconda3\\ForestTrade\\log\\trade_log.txt')
# filename = os.path.abspath(os.path.realpath(filename))
# print (filename)
# readFile(filename)
# =============================================================================
