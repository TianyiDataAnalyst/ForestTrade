# -*- coding: utf-8 -*-
"""
Created on Wed May 13 08:22:41 2020

@author: gutia
"""

# The Tic Tac Toe game
import numpy as np

class TicTacToe():
  def __init__(self, state=None):
    self.reset()
    if state is not None:
      self.state = state

  def reset(self):
    self.done = False
    self.state = [0]*11
    self.state[-1] = 1
    return self.state
   
  class observation_space():
    shape = (11,)

  class action_space():
    n = 9

  def render(self):
    print("turn %d" % self.state[-1])
    print(np.array(self.state[0:9]).reshape(3,3))
    
  def value(self, s):
    ret = 0
    for turn in [-1, 1]:
      for i in range(3):
        if all([x==turn for x in s[3*i:3*i+3]]):
          ret = turn
        if all([x==turn for x in [s[i], s[3+i], s[6+i]]]):
          ret = turn
      if all([x==turn for x in [s[0], s[4], s[8]]]):
        ret = turn
      if all([x==turn for x in [s[2], s[4], s[6]]]):
        ret = turn
    # NOTE: this is not the value, the state may be won
    return ret*s[-1]
  
  def dynamics(self, s, act):
    rew = 0
    s = s.copy()
    if s[act] != 0 or s[-2] != 0:
      # don't move in taken spots or in finished games
      rew = -10
    else:
      s[act] = s[-1]
      rew += self.value(s)
    if s[-2] != 0:
      rew = 0
    else:
      s[-2] = self.value(s)
    s[-1] = -s[-1]
    return rew, s
  
  def step(self, act):
    rew, self.state = self.dynamics(self.state, act)
    if rew != 0:
      self.done = True
    if np.all(np.array(self.state[0:9]) != 0):
      self.done = True
    return self.state, rew, self.done, None
  
# Play a quick round
env = TicTacToe()
print(env.reset())
print(env.step(4))
print(env.step(0))
print(env.step(3))
print(env.step(1))
print(env.step(6))
print(env.step(2))
print(env.state[-1], env.value(env.state))



#==============================================================================================
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
from ForestTrade.muzero.mcts import mcts_search, print_tree
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

#===========================================================================================
# computer can play against itself...and tie!

gg = TicTacToe()
done = False
while not done:
  policy, node = mcts_search(mm, gg.state, 2000)
  print(policy)
  act = np.random.choice(list(range(len(policy))), p=policy)
  print(act)
  _, _, done, _ = gg.step(act)
  gg.render()
  
  
  #========================================================================================
  # Now we try to learn a model, and things work less well

1