import numpy as np

class State:
    def __init__(self, board= None, player = 1) -> None:
        self.board = board
        self.player = player
        self.action : tuple[int, int] = None

    def get_opponent (self):
        if self.player == 1:
            return 2
        else:
            return 1

    def switch_player(self):
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    def score (self, player = 1) -> tuple[int, int]:
        if player == 1:
            opponent = 2
        else:
            opponent = 1

        player_score = np.count_nonzero(self.board == player)
        opponent_score = np.count_nonzero(self.board == opponent)
        return player_score, opponent_score

    def __eq__(self, other) ->bool:
        b1 = np.equal(self.board, other.board).all()
        b2 = self.player == other.player
        return np.equal(self.board, other.board).all() and self.player == other.player

    def __hash__(self) -> int:
        return hash(repr(self.board) + self.player)
    
    def copy (self):
        newBoard = np.copy(self.board)
        return State(board=newBoard, player=self.player)
    