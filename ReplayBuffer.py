from collections import deque
import random
import torch

capacity = 10000

class ReplayBuffer:
    def __init__(self, capacity= 10000) -> None:
        self.buffer = deque(maxlen=capacity)

    def push (self, state_tensor, reward_tensor, next_state_tensor, done):
        self.buffer.append((state_tensor, reward_tensor, next_state_tensor, done))
            
    def sample (self, batch_size):
        state_tensors, reward_tensors, next_state_tensors, dones = zip(*random.sample(self.buffer, batch_size))
        states = torch.vstack(state_tensors)
        rewards = torch.vstack(reward_tensors)
        next_states = torch.vstack(next_state_tensors)
        done_tensor = torch.tensor(dones).long().reshape(-1,1)
        
        return states, rewards, next_states, done_tensor

    def __len__(self):
        return len(self.buffer)

