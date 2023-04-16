from Reversi import Reversi
from MinMaxAgent import MinMaxAgent
from MinMaxAgent2 import MinMaxAgent2
from AlphBetaAgent import AlphaBetaAgent
from DQN import DQN
from DQNAgent import DQNAgent
from State import State
from RandomAgent import RandomAgent
from FixAgent import FixAgent
from FixAgent2 import FixAgent2
import torch

environment = Reversi()
# player1 = MinMaxAgent(player = 1,depth = 3, environment=environment)
# player2 = MinMaxAgent(player = 2,depth = 3, environment=environment)
# player1 = MinMaxAgent2(player = 1,depth = 3, environment=environment)
# player2 = MinMaxAgent2(player = 2,depth = 3, environment=environment)
player1 = AlphaBetaAgent(player = 1,depth = 3, environment=environment)
# player2 = AlphaBetaAgent(player = 2,depth = 3, environment=environment)
# player1 = RandomAgent(environment)
# player2 = RandomAgent(environment)
# player1 = FixAgent(environment, player=1)
# player2 = FixAgent(environment, player=2)
# player1 = FixAgent2(environment, player=1)
# player2 = FixAgent2(environment, player=2)

# file='Reversi/Data/DQN_Model_AI_AI_best_eval_W.pth'
file='Reversi/Data/DQN_Model_AI_AI_best_eval_B.pth'
# model = DQN(environment)
model = torch.load(file)
# player1 = DQNAgent(model, player=1, train=False)
player2 = DQNAgent(model, player=2, train=False)

# fileWhite='DQN_model_White.pth'
# model_White = torch.load(fileWhite)
# player1 = DQNAgent(model_White, player=1, train=False)

# fileBlack='DQN_Model_Black.pth'
# model_Black = torch.load(fileBlack)
# player2 = DQNAgent(model_Black, player=2, train=False)


def main ():
    player = player1
    player1_win = 0
    player2_win = 0
    games = 0
    while games < 1:
        action = player.get_Action(state=environment.state)
        environment.move(action, environment.state)
        player = switchPlayers(player)
        if environment.is_end_of_game(environment.state):
            score1, score2 = environment.state.score()
            print ("player 1: score = ", score1, "player 2: score = ", score2 )
            if score1 > score2:
                player1_win += 1
            else:
                player2_win += 1
            environment.state = environment.get_init_state()
            player = player1
            games += 1
            print (f"Game no.: {games}, score: {player1_win, player2_win}",end="\r")
    print("End of game")
    
    print ("player 1: wins = ", player1_win/games)
    print ("player 2: wins = ", player2_win/games)
    

def switchPlayers(player):
    if player == player1:
       return player2
    else:
        return player1

if __name__ == '__main__':
    main()
    
