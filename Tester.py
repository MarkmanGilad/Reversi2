from Reversi import Reversi
from MinMaxAgent import MinMaxAgent
from MinMaxAgent2 import MinMaxAgent2
from AlphBetaAgent import AlphaBetaAgent
from DQN import DQN
from DQNAgent import DQNAgent
from State import State
from RandomAgent import RandomAgent

file='DQN_Model.pth'
environment = Reversi()
# player1 = Human_Agent(player=1)
# player2 = Human_Agent(player=2)
# player1 = MinMaxAgent(player = 1,depth = 3, environment=environment)
# player2 = MinMaxAgent(player = 2,depth = 3, environment=environment)
# player1 = MinMaxAgent2(player = 1,depth = 3, environment=environment)
# player2 = MinMaxAgent2(player = 2,depth = 3, environment=environment)
# player1 = AlphaBetaAgent(player = 1,depth = 3, environment=environment)
# player2 = AlphaBetaAgent(player = 2,depth = 4, environment=environment)

model = DQN(environment)
player1 = DQNAgent(model, player=1, train=False)
player1.loadModel(file)
# player2 = DQNAgent(model, player=2, train=False)
# player2.loadModel(file)
# player1 = RandomAgent(environment)
player2 = RandomAgent(environment)


def main ():
    player = player1
    player1_win = 0
    player2_win = 0
    games = 0
    while games < 100:
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
            games += 1
            print ("Game no.: ", games)
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
    
