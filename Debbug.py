from State import State
from Reversi import Reversi
import numpy as np
from AlphBetaAgent import AlphaBetaAgent

env = Reversi()

player = AlphaBetaAgent(player=1, depth=3, environment=env)
board = np.array([[1,1,1,1,1,2,2,2],
                  [0,1,1,1,2,2,2,2],
                  [2,1,1,1,2,2,2,2],
                  [2,1,1,2,2,2,2,2],
                  [2,1,1,2,2,2,2,2],
                  [2,1,1,2,2,2,2,2],
                  [1,2,2,2,2,2,2,2],
                  [2,0,0,2,2,2,2,2]
                  ]
                 , dtype=float)
print (board)
state = State(board=board, player=2.0)
action = player.get_state_action(state=state)
print(action)
