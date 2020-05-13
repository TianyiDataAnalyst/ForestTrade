# -*- coding: utf-8 -*-
"""
Created on Sun May 10 23:44:27 2020

@author: gutia
"""
"""
Implementing MuZero without being a savage and spam programming

Three parts:

1. Model(f,g,h)
  representation:  s_0 = h(o_1, ..., o_t)
  dynamics:        r_k, s_k = g(s_km1, a_k)
  prediction:      p_k, v_k = f(s_k)

2. Acting
  MCTS algorithm

3. Training
  Experience replay
  
"""  

    
#https://www.tensorflow.org/install/source_windows

import os  
os.chdir("C:\\Users\\gutia\\Anaconda3")
os. getcwd() 
