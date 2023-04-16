import numpy as np
from Reversi import Reversi
from State import State
import random
import torch


class FixAgent2:
    def __init__(self, env, player = 1) -> None:
        self.env  = env
        self.player = player

    def value(self, state: State):
        v = np.array([[120, -50, 10, 20, 20, 10, -50, 120], 
             [-50, -50, 1, 1, 1, 1, -50, -50],
             [10, 1, 10, 2, 2, 10, 1, 10],
             [20,1,2,5,5,2,1,20],
             [20,1,2,5,5,2,1,20],
             [10, 1, 10, 2, 2, 10, 1, 10],
             [-50, -50, 1, 1, 1, 1, -50, -50],
             [120, -50, 10, 20, 20, 10, -50, 120]])
        board = state.board
        score1 = ((board % 2) * v).sum()
        score2 = ((board // 2) * v).sum()
        if self.player == 1:
             return score1 - score2
        else:
             return score2 - score1
              

    def get_Action (self, event = None, graphics=None, state: State = None, epoch = 0):
        next_states, legal_actions = self.env.get_all_next_states(state)
        values = []
        for next_state in next_states:
                values.append(self.value(next_state))
        maxIndex = values.index(max(values))
        return legal_actions[maxIndex]

    def get_state_action(self, event = None, graphics=None, state: State = None, epoch = 0, train = False):
        next_states, legal_actions = self.env.get_all_next_states(state)
        values = []
        for next_state in next_states:
                values.append(self.value(next_state))
        maxIndex = values.index(max(values))
        return next_states[maxIndex].toTensor(),legal_actions[maxIndex]