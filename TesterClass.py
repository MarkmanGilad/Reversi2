import torch
from RandomAgent import RandomAgent
from FixAgent import FixAgent
from DQN import DQN
from DQNAgent import DQNAgent
from Reversi import Reversi

class tester:

    def __init__(self, env, player1, player2) -> None:
        self.env = env
        self.player1 = player1
        self.player2 = player2
        

    def test (self, games_num):
        env = self.env
        player = self.player1
        player1_win = 0
        player2_win = 0
        games = 0
        while games < games_num:
            action = player.get_Action(state=env.state)
            env.move(action, env.state)
            player = self.switchPlayers(player)
            if env.is_end_of_game(env.state):
                score1, score2 = env.state.score()
                if score1 > score2:
                    player1_win += 1
                else:
                    player2_win += 1
                env.state = env.get_init_state()
                games += 1
                player = self.player1
        return player1_win, player2_win        

    def switchPlayers(self, player):
        if player == self.player1:
            return self.player2
        else:
            return self.player1

if __name__ == '__main__':
    env = Reversi()
    # player2 = FixAgent(env, player=2)
    player2 = RandomAgent(env)
    file = 'DQN_Model_W_Fix3.pth'
    model = torch.load(file)
    player1 = DQNAgent(model, player=1, train=False)
    test = tester(env,player1, player2)
    print(test.test())
    
