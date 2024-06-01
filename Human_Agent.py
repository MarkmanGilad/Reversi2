import pygame
from Graphics import *
import time

class Human_Agent:

    def __init__(self, player: int) -> None:
        self.player = player

    def get_Action (self, events= None, graphics: Graphics = None, state = None, train = False):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row_col = graphics.calc_row_col(pos)
                time.sleep(0.2) 
                return row_col
            else:
                return None