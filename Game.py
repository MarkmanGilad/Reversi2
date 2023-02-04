import pygame
from Graphics import *
from Reversi import Reversi
from Human_Agent import Human_Agent
from MinMaxAgent import MinMaxAgent

>>>>>>> 33c267c72b988bd57bc7d00460870a25cc51adb8
import time

FPS = 60

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Reversi')
environment = Reversi()
graphics = Graphics(win, board = environment.state.board)
player1 = Human_Agent(player=1)
player2 = Human_Agent(player=2)
# player1 = minMaxAgent(player = 1,depth = 3)
# player1 = AlphaBetaAgent(player = 1,depth = 3)
# player2 = MinMaxAgent(player = 2,depth = 3, environment=environment)
# player2 = AlphaBetaAgent(player = 2,depth = 4)


def main ():
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
                # graphics.blink(action, GREEN)
                player = switchPlayers(player)
            else:
                graphics.blink(action, RED)

        graphics.draw()
        pygame.display.update()
        if environment.is_end_of_game(environment.state):
            run = False
    
    # time.sleep(2)
    pygame.quit()
    print("End of game")
    score1, score2 = environment.state.score()
    print ("player 1: score = ", score1)
    print ("player 2: score = ", score2)


def switchPlayers(player):
    if player == player1:
       return player2
    else:
        return player1

if __name__ == '__main__':
    main()
    
