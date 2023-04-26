from Reversi import Reversi
from State import State
MAXSCORE = 1000

class MinMaxAgent2:

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

    def get_Action(self, event, graphics, state: State):
        value, bestAction = self.minMax(state)
        return bestAction

    def minMax(self, state:State):
        visited = set()
        depth = 0
        return self.max_value(state, visited, depth)
        
    def max_value (self, state:State, visited:set, depth):
        
        value = -MAXSCORE

        # stop state
        if depth == self.depth or self.environment.is_end_of_game(state):
            value = self.evaluate(state)
            return value, state.action
        
        # start recursion
        bestAction = None
        legal_actions = self.environment.get_legal_actions(state)
        for action in legal_actions:
            newState = self.environment.get_next_state(action, state)
            if newState not in visited:
                visited.add(newState)
                newValue, newAction = self.min_value(newState, visited,  depth + 1)
                if newValue > value:
                    value = newValue
                    bestAction = action

        if bestAction:
            return value, bestAction 
        else:
            return MAXSCORE, bestAction
  

    def min_value (self, state:State, visited:set, depth):
        
        value = MAXSCORE

        # stop state
        if depth == self.depth or self.environment.is_end_of_game(state):
            value = self.evaluate(state)
            return value, state.action
        
        # start recursion
        bestAction = None
        legal_actions = self.environment.get_legal_actions(state)
        for action in legal_actions:
            newState = self.environment.get_next_state(action, state)
            if newState not in visited:
                visited.add(newState)
                newValue, newAction = self.max_value(newState, visited,  depth + 1)
                if newValue < value:
                    value = newValue
                    bestAction = action

        if bestAction:
            return value, bestAction 
        else:
            return -MAXSCORE, bestAction
 

