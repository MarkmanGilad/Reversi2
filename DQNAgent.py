import numpy as np
from Reversi import Reversi
from State import State
from DQN import *

class DQNAgent:
    def __init__(self, model, player, env = None) -> None:
        self.model = model
        self.player = player
        self.env = env

    def get_Action (self, event = None, graphics=None, state: State = None):
        _, action = self.model.act(state)
        return action

