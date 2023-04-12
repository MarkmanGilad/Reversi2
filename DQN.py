import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Reversi import Reversi


# Parameters

input_size = 65 # state: board = 8 * 8 + palyer = 1
layer1 = 128
layer2 = 64
output_size = 1 # V(state)
# epochs = 1000
# batch_size = 64
gamma = 1 

# epsilon Greedy
epsilon_start = 1.0
epsilon_final = 0.01
epsiln_decay = 500000

MSELoss = nn.MSELoss()

class DQN (nn.Module):
    def __init__(self, env) -> None:
        super().__init__()
        self.env = env 
        if torch.cuda.is_available:
            self.device = torch.device('cpu') # 'cuda'
        else:
            self.device = torch.device('cpu')
        
        self.linear1 = nn.Linear(input_size, layer1, device=self.device)
        self.linear2 = nn.Linear(layer1, layer2, device=self.device)
        self.output = nn.Linear(layer2, output_size, device=self.device)
        
    def forward (self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        x = self.output(x)
        return x

    def act (self, state, epoch = 1, train = True):
        epsilon = epsilon_greedy(epoch)
        rnd = random.random()
        if rnd < epsilon and train:
            actions = self.env.get_legal_actions(state)
            action = random.choice(actions)
            next_state = self.env.get_next_state(action, state)
            return next_state.toTensor(self.device), action
        
        next_states, legal_actions = self.env.get_all_next_states (state)
        states_tensor = self.env.toTensor(next_states, self.device)
        q_values = self.forward(states_tensor)
        maxIndex = torch.argmax(q_values)
        return states_tensor[maxIndex], legal_actions[maxIndex]
    

    def loss (self, Q_value, rewards, Q_next_Values, Dones ):
        Q_new = rewards + gamma * Q_next_Values * (1- Dones)
        return MSELoss(Q_value, Q_new)

def epsilon_greedy(epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay):
    res = final + (start - final) * math.exp(-1 * epoch/decay)
    return res
        