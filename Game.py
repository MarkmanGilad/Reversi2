import pygame
from Graphics import *
from Reversi import Reversi
from Human_Agent import Human_Agent
from MinMaxAgent import MinMaxAgent
from MinMaxAgent2 import MinMaxAgent2
from AlphBetaAgent import AlphaBetaAgent
from DQN import DQN
from DQNAgent import DQNAgent
from State import State
from RandomAgent import RandomAgent
from FixAgent import FixAgent
from FixAgent2 import FixAgent2
import time
import torch

FPS = 60
file='Data/DQN_Model_AB_3.pth'
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Reversi')
environment = Reversi()
graphics = Graphics(win, board = environment.state.board)
player1 = Human_Agent(player=1)
player2 = Human_Agent(player=2)
# player1 = MinMaxAgent(player = 1,depth = 3, environment=environment)
# player2 = MinMaxAgent(player = 2,depth = 3, environment=environment)
# player1 = MinMaxAgent2(player = 1,depth = 3, environment=environment)
# player2 = MinMaxAgent2(player = 2,depth = 3, environment=environment)
# player1 = AlphaBetaAgent(player = 1,depth = 3, environment=environment)
# player2 = AlphaBetaAgent(player = 2,depth = 3, environment=environment)
# player1 = RandomAgent(environment)
# player2 = RandomAgent(environment)
# player1 = FixAgent(environment, player=1)
# player2 = FixAgent(environment, player=2, train=True)
# player1 = FixAgent2(environment, player=1, train=True)
# player2 = FixAgent2(environment, player=2)

# model = DQN(environment)
# model = torch.load(file)
# player1 = DQNAgent(model, player=1, train=False)
# player2 = DQNAgent(model, player=2, train=False)

firstPlayer = player1

def main ():
    start = time.time()
    run = True
    clock = pygame.time.Clock()
    graphics.draw()
    player = player1
    
    while(run):
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
               run = False
               
        action = player.get_Action(event, graphics, environment.state)
        if action:
            if (environment.move(action, environment.state)):
                graphics.blink(action, GREEN)
                player = switchPlayers(player)
            else:
                graphics.blink(action, RED)
        
        graphics.draw()
        pygame.display.update()
        if environment.is_end_of_game(environment.state):
            # run = False
            score1, score2 = environment.state.score()
            print ("player 1: score = ", score1)
            print ("player 2: score = ", score2)
            print(environment.state.reward(1))
            time.sleep(2)
            environment.state = environment.get_init_state()
            graphics.board = environment.state.board
            player = firstPlayer
            graphics.draw()
    time.sleep(2) 
    pygame.quit()
    print("End of game")
    score1, score2 = environment.state.score()
    print ("player 1: score = ", score1)
    print ("player 2: score = ", score2)
    # print (time.time() - start)


def switchPlayers(player):
    if player == player1:
       return player2
    else:
        return player1

if __name__ == '__main__':
    main()
    
