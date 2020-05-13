# -*- coding: utf-8 -*-
"""
Created on Wed May 13 08:24:28 2020

@author: gutia
"""

# a mock representation, dynamics, and prediction function
class MockModel():
  def ht(self,s):
    return s
  def gt(self, s, a):
    #print(s, a)
    return env.dynamics(s,a)
  def ft(self,s):
    #print(s, env.value(s))
    return np.array([1/9]*9), env.value(s)

# unit tests for the MCTS!
from muzero.mcts import mcts_search, print_tree
mm = MockModel()
obs = [1, -1, 1, -1, 1, -1, 0, 0, 0,  0,1]
policy, node = mcts_search(mm, obs, 1000)
print(policy)
act = np.random.choice(list(range(len(policy))), p=policy)
assert act == 8 or act == 6
obs = [-1, -1, 0, 1, 1, 0, 1, 0, 0,  0,-1]
policy, node = mcts_search(mm, obs, 1000)
print(policy)
act = np.random.choice(list(range(len(policy))), p=policy)
assert act == 2
obs = [1,0,0,1,-1,0,-1,0,0,   0,1]
policy, node = mcts_search(mm, obs, 1000)
print(policy)
act = np.random.choice(list(range(len(policy))), p=policy)
assert act == 2
obs = [0,1,-1,0,1,0,0,0,0,  0,-1]
policy, node = mcts_search(mm, obs, 1000)
print(policy)
act = np.random.choice(list(range(len(policy))), p=policy)
assert act == 7
obs = [0,0,0, 0,-1,0, 1,-1,1,  0,1]
policy, node = mcts_search(mm, obs, 1000)
print(policy)
act = np.random.choice(list(range(len(policy))), p=policy)
assert act == 1