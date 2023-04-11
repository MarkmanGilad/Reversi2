import numpy as np
from Reversi import Reversi
from State import State
import random


class RandomAgent:
    def __init__(self, env) -> None:
        self.env = env

    def get_Action (self, event = None, graphics=None, state: State = None, epoch = 0):
            actions = self.env.get_legal_actions(state)
            action = random.choice(actions)
            # next_state = self.env.get_next_state(action, state)
            return action
