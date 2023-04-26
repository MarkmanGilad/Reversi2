import numpy as np
from Reversi import Reversi
from State import State
import random
import torch


class FixAgent:
    def __init__(self, env, player = 1, train = False) -> None:
        self.env  = env
        self.player = player
        self.train = train

    def value(self, state: State):
        v = np.array([[100, -25, 10, 5, 5, 10, -25, 100], 
                           [-25, -25, 2, 2, 2, 2, -25, -25],
                           [10, 2, 5, 1, 1, 5, 2, 10],
                           [5,2,1,2,2,1,2,5],
                           [5,2,1,2,2,1,2,5],
                           [10, 2, 5, 1, 1, 5, 2, 10],
                           [-25, -25, 2, 2, 2, 2, -25, -25],
                           [100, -25, 10, 5, 5, 10, -25, 100]])
        board = state.board
        score1 = ((board % 2) * v).sum()
        score2 = ((board // 2) * v).sum()
        if self.player == 1:
             return score1 - score2
        else:
             return score2 - score1
              

    def get_Action (self, event = None, graphics=None, state: State = None, epoch = 0):
        next_states, legal_actions = self.env.get_all_next_states(state)
        if self.train and random.random() < 0.1:
             return random.choice(legal_actions)
        values = []
        for next_state in next_states:
                values.append(self.value(next_state))
        maxIndex = values.index(max(values))
        return legal_actions[maxIndex]

    def get_state_action(self, event = None, graphics=None, state: State = None, epoch = 0):
        next_states, legal_actions = self.env.get_all_next_states(state)
        if self.train and random.random() < 0.1:
             index = random.randint(0,len(next_states)-1)
             return next_states[index].toTensor(),legal_actions[index]
        values = []
        for next_state in next_states:
                values.append(self.value(next_state))
        maxIndex = values.index(max(values))
        return next_states[maxIndex].toTensor(),legal_actions[maxIndex]