import numpy as np
import torch
from Reversi import Reversi
from State import State
from DQN import *

class DQNAgent:
    def __init__(self, model, player, env = None, train = True) -> None:
        self.model = model
        self.player = player
        self.env = env
        self.train = train

    def get_Action (self, event = None, graphics=None, state: State = None, epoch = 0):
        if not self.train:
            with torch.no_grad():
                _, action = self.model.act(state, epoch, train=False)
        else:
            _, action = self.model.act(state, epoch, train = True)
        return action

    def get_state_action (self, event = None, graphics=None, state: State = None, epoch = 0):
        return self.model.act(state, epoch)
        

    def loadModel (self, file):
        self.model = torch.load(file)
        