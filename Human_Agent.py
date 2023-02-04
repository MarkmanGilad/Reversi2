import pygame
from Graphics import *

class Human_Agent:

    def __init__(self, player: int) -> None:
        self.player = player

    def get_Action (self, event= None, graphics: Graphics = None, state = None):
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            row_col = graphics.calc_row_col(pos) 
            return row_col
        else:
            return None