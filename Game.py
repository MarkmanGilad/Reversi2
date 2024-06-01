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
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Reversi')
player1 = None
player2 = None
environment = Reversi()

def main (p1, p2):
    global player1, player2
    player1=p1
    player2=p2   
    
    graphics = Graphics(win, board = environment.state.board)
    
    file='Data/DQN_Model_AB_3.pth'
    firstPlayer = player1
    start = time.time()
    run = True
    clock = pygame.time.Clock()
    graphics.draw()
    player = player1
    
    while(run):
        clock.tick(FPS)
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
               run = False
               
        action = player.get_Action(events, graphics, environment.state)
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


def GUI ():
    global player1, player2
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

    colors = [['blue', 'gray', 'gray', 'gray'], ['blue', 'gray', 'gray', 'gray']]
    player1_chosen = 0
    player2_chosen = 0
    clock = pygame.time.Clock()
    run = True
    while(run):
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
               run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if 300<pos[0]<500 and 500<pos[1]<540:
                    main(player1, player2) 
                if 100<pos[0]<300 and 200<pos[1]<240:
                    player1 = Human_Agent(player=1)
                    player1_chosen=0
                if 500<pos[0]<800 and 200<pos[1]<240:
                    player1 = Human_Agent(player=1)
                    player2_chosen=0
                if 100<pos[0]<300 and 250<pos[1]<290:
                    player1 = MinMaxAgent(player = 1,depth = 3, environment=environment)
                    player1_chosen=1
                if 500<pos[0]<800 and 250<pos[1]<290:
                    player2 = MinMaxAgent(player = 2,depth = 3, environment=environment)
                    player2_chosen=1



        colors = [['gray', 'gray', 'gray', 'gray'], ['gray', 'gray', 'gray', 'gray']]
        colors[0][player1_chosen]='BLUE'
        colors[1][player2_chosen]='BLUE'




        win.fill('LightGray')
        write(win, "Space Invaders", pos=(300, 50), color=BLACK, background_color=None)

        write(win, 'Player 1',(150,150),color=BLACK)
        pygame.draw.rect(win, colors[0][0], (100,200,200,40))
        write(win, 'Human', (120,200),color=BLACK)
        pygame.draw.rect(win, colors[0][1], (100,250,200,40))
        write(win, 'Min_Max', (120,250),color=BLACK)
        pygame.draw.rect(win, colors[0][2], (100,300,200,40))
        write(win, 'Alpha_Beta', (120,300),color=BLACK)
        pygame.draw.rect(win, colors[0][3], (100,350,200,40))
        write(win, 'DQN', (120,350),color=BLACK)

        write(win, 'Player 2',(550,150),color=BLACK)
        pygame.draw.rect(win, colors[1][0], (500,200,200,40))
        write(win, 'Human', (520,200),color=BLACK)
        pygame.draw.rect(win, colors[1][1], (500,250,200,40))
        write(win, 'Min_Max', (520,250),color=BLACK)
        pygame.draw.rect(win, colors[1][2], (500,300,200,40))
        write(win, 'Alpha_Beta', (520,300),color=BLACK)
        pygame.draw.rect(win, colors[1][3], (500,350,200,40))
        write(win, 'DQN', (520,350),color=BLACK)

        
        pygame.draw.rect(win, 'gray', (300,500,200,40))
        write(win, 'Play', (350,500),color=BLACK)


        pygame.display.update()

    pygame.quit()

def write (surface, text, pos = (50, 20), color = BLACK, background_color = None):
    font = pygame.font.SysFont("arial", 36)
    text_surface = font.render(text, True, color, background_color)
    surface.blit(text_surface, pos)


if __name__ == '__main__':
    GUI()
    
