import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from game import FleetBattleEnv

from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 32
LR = 0.01

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0
        self.memory = deque(maxlen = MAX_MEMORY)
        self.model = None
        self.trainer = None
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def train_long_memory(self):
        ...
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
    def get_action(self, state):
        ...

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = FleetBattleEnv()
    old_state = game.reset()  # so that we get the current state
    while True:
        final_move = agent.get_action(old_state)
        
        new_state, done, reward, score = game.step(final_move)
        
        agent.train_short_memory(old_state, final_move, reward, new_state, done)
        
        # agent.remember(state_old, final_move, reward, state_new, done)
        
        # if done:
        #     game.reset()
        #     agent.n_games += 1
        #     agent.train_long_memory()
            
        #     if score > record:
        #         record = score
        #         # agent.model.save()
            
        #     print('Game', agent.n_games, 'Score', score, 'Record', record)
            
        #     # plot_scores.append(score)
        #     # total_score += score
        #     # mean_score = total_score / agent.n_games
        #     # plot_mean_scores.append(mean_score)
        #     # plot(plot_scores, plot_mean_scores)
        
        old_state = new_state

if __name__ == '__main__':
    train()
