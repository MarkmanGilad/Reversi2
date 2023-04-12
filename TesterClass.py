import torch

class tester:

    def __init__(self, env, player1, player2) -> None:
        self.env = env
        self.player1 = player1
        self.player2 = player2

    def test (self):
        env = self.env
        player = self.player1
        player1_win = 0
        player2_win = 0
        games = 0
        while games < 100:
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
        
        return player1_win, player2_win        

    def switchPlayers(self, player):
        if player == self.player1:
            return self.player2
        else:
            return self.player1

if __name__ == '__main__':
    tester.test()
    
