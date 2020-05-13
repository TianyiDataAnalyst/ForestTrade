# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:03:37 2020

@author: gutia
"""
# this is MuZero!
# https://arxiv.org/abs/1911.08265
# It works, but only for CartPole
# Would be happy if it worked for Follower, LunarLander, or Acrobot
# Also, while it's written, it's not using mcts.
# Replace naive_search(which tries n**K) with mcts_search to use
#https://github.com/geohot/ai-notebooks/blob/master/muzero_cartpole_v3.ipynb


import tensorflow as tf
import numpy as np
import gym
from tqdm import tqdm, trange
import os,sys

sys.path.append(os.getcwd())

#Populating the interactive namespace from numpy and matplotlib

# Make Follower work! Will give interview to anyone who does.
from  ForestTrade.muzero.follower import Follower
#env = Follower()
env = gym.make("CartPole-v0")
#env = gym.make("MountainCar-v0")
#env = gym.make("LunarLander-v2")
#env = gym.make("Acrobot-v1")

from  ForestTrade.muzero.model import MuModel
m = MuModel(env.observation_space.shape, env.action_space.n, s_dim=128, K=3, lr=0.001)
print(env.observation_space.shape, env.action_space.n)

from  ForestTrade.muzero.game import Game, ReplayBuffer
from  ForestTrade.muzero.mcts import naive_search, mcts_search
replay_buffer = ReplayBuffer(50, 128, m.K)
rews = []