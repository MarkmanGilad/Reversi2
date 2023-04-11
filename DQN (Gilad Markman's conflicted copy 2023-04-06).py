import numpy as nn
import torch 
import torch.nn as nn
import torch.optim as optim


class DQN (nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.input = nn.Linear()