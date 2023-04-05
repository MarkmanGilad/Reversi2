import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Reversi import Reversi


# Parameters
learning_rate = 0.01
input_size = 65 # state: board = 8 * 8 + palyer = 1
layer1 = 128
layer2 = 64
output_size = 1 # V(state)
epoch = 100
batch_size = 64
gamma = 0.99 

# epsilon Greedy
epsilon_start = 1.0
epsilon_final = 0.01
epsiln_decay = 500


class DQN (nn.Module):
    def __init__(self, env) -> None:
        super().__init__()
        self.env = env 
        if torch.cuda.is_available:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.linear1 = nn.Linear(input_size, layer1, device=self.device)
        self.linear2 = nn.Linear(layer1, layer2, device=self.device)
        self.output = nn.Linear(layer2, output_size, device=self.device)
        
    def forward (self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.output(x)
        return x
    
    def act (self, state, epsilon = 0):
        if random.random() < epsilon:
            actions = self.env.get_legal_actions(state)
            action = random.choice(actions)
            next_state = self.env.get_next_state(action, state)
            return next_state.toTensor(self.device), action
        
        next_states, legal_actions = self.env.get_all_next_states (state)
        states_tensor = self.env.toTensor(next_states, self.device)
        q_values = self.forward(states_tensor)
        maxIndex = torch.argmax(q_values)
        return states_tensor[maxIndex], legal_actions[maxIndex]
    

def epsilon_greedy(epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay):
    return final + (start - final) * math.exp(-1 * epoch/decay)
        