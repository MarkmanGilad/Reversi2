from Reversi import Reversi
from State import State
from DQN import *
from DQNAgent import DQNAgent
from ReplayBuffer import ReplayBuffer
from FixAgent import FixAgent
from RandomAgent import RandomAgent
from TesterClass import tester
import torch

buffer = ReplayBuffer()
epochs = 1500000
learning_rate = 0.1
batch_size = 64
env = Reversi()
testEnv = Reversi()

model = DQN(env)
file='DQN_Model_W_Fix1.pth'
model = torch.load(file)
player1 = DQNAgent(model, player=1)
playerFix = FixAgent(env, player=2)
playerRand = RandomAgent(env)
player2 = playerFix

test = tester(testEnv, player1, playerFix)

def main ():

    # init optimizer
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    state = env.state.toTensor(device=model.device)

    for epoch in range(epochs):
        if random.random() < 0.05:
            player2 = playerRand
        else:
            player2 = playerFix
        
        reward = 0    

        state_Q, _ = player1.get_state_action(state=State.tensorToState(state), epoch=0)

        final_state_s = State.tensorToState(state_Q)
        if env.is_end_of_game(final_state_s):
            state = end_of_game(final_state_s, state_Q, state_Q)
            continue

        next_state, _ = player2.get_state_action(state=State.tensorToState(state_Q), epoch=0)
        final_state_s = State.tensorToState(next_state)
        if env.is_end_of_game(final_state_s):
            state = end_of_game(final_state_s, state_Q, next_state_Q)
            continue

        next_state_Q, _ = player1.get_state_action(state=State.tensorToState(next_state), epoch=0)
        final_state_s = State.tensorToState(next_state_Q)
        if env.is_end_of_game(final_state_s):
            state = end_of_game(final_state_s, state_Q, next_state_Q)
            continue
    
        final_state, _ = player2.get_state_action(state=State.tensorToState(next_state), epoch=0)
        final_state_s = State.tensorToState(final_state)
        if env.is_end_of_game(final_state_s): 
            state = end_of_game(final_state_s, state_Q, next_state_Q)
            continue
       
        buffer.push(state_Q, torch.tensor(reward, dtype=torch.float32), next_state_Q, False)
      
        state = next_state
        
        if epoch % 100 == 0:
            print (epoch, end="\r")
        
        if epoch % 25000 == 0:
            torch.save(model, file)        

        if len(buffer) < 500:
            continue

        states_Q, rewards, next_states_Q, dones = buffer.sample(batch_size)
        
        # zero wights
        optim.zero_grad()

        # forward
        Q_value = model.forward(states_Q)
        with torch.no_grad():
            Q_next_value = model.forward(next_states_Q)

        #backward()
        loss = model.loss(Q_value, rewards, Q_next_value, dones)
        loss.backward()

        if epoch % 1000 == 0:
            print (f'epoch: {epoch}, loss: {loss}', end="\r")

        if epoch % 10000 == 0:
            print (f'test: {test.test()}')

        # update wights
        optim.step()

    print (len(buffer))
    
    torch.save(model, file)

def end_of_game (final_state_s, state_Q, next_state_Q):
    reward = final_state_s.reward(player1.player)
    reward = torch.tensor(reward, dtype=torch.float32)
    buffer.push(state_Q, reward, next_state_Q, True)
    return env.get_init_state().toTensor()

if __name__ == '__main__':
    main()
