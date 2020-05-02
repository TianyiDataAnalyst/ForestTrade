# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:46:00 2020

@author: gutia
"""
import os
os.path.abspath(os.getcwd())

#change file directory to upper level
os.chdir(os.path.dirname(os.getcwd()))


from pathlib import Path
print("File      Path:", Path('C:\\Users\\gutia\\Anaconda3\\ForestTrade\\current_directory.py').absolute())
print("Directory Path:", Path().absolute())