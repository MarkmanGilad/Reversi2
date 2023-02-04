from Reversi import Reversi
from State import State
MAXSCORE = 1000

class MinMaxAgent:

    def __init__(self, player, depth = 2, environment: Reversi = None):
        self.player = player
        if self.player == 1:
            self.opponent = 2
        else:
            self.opponent = 1
        self.depth = depth
        self.environment : Reversi = environment

    def evaluate (self, gameState : State):
        player_score, opponent_score = gameState.score(player = self.player)
        score =  player_score - opponent_score
        
        for row in range(0, 7):
            for col in (0, 7):
                if gameState.board[row][col] == self.player:
                    score += 5
                elif gameState.board[row][col] == self.opponent:
                    score -= 5
        
        for row in (0, 7):
            for col in range (0, 7):
                if gameState.board[row][col] == self.player:
                    score += 5
                elif gameState.board[row][col] == self.opponent:
                    score -= 5
        
        for row in (0,7):
            for col in (0,7):
                if gameState.board[row][col] == self.player:
                    score += 10
                elif gameState.board[row][col] == self.opponent:
                    score -= 10

        return score

    def get_Action(self, event, graphics, gameState):
        reached = []
        value, bestAction = self.minMax(gameState, reached, 0)
        return bestAction

    def minMax(self, gameState, reached, depth):
        if self.player == gameState.player:
            value = -MAXSCORE
        else:
            value = MAXSCORE

        # stop state
        if depth == self.depth or self.environment.is_end_of_game(gameState):
            value = self.evaluate(gameState)
            return value, None
        
        bestAction = None
        legal_actions = self.environment.get_legal_actions(gameState)
        # start recursion
        for action in legal_actions:
            newGameState = self.environment.get_next_state(action, gameState)
            if newGameState not in reached:
                reached.append(newGameState)
                if self.player == gameState.player:         # maxNode - agent
                    newValue, newAction = self.minMax(newGameState, reached,  depth + 1)
                    if newValue > value:
                        value = newValue
                        bestAction = action
                else:                       # minNode - opponent
                    newValue, newAction = self.minMax(newGameState, reached,  depth + 1)
                    if newValue < value:
                        value = newValue
                        bestAction = action

        return value, bestAction 
