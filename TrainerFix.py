from Reversi import Reversi
from State import State
from DQN import *
from DQNAgent import DQNAgent
from ReplayBuffer import ReplayBuffer
from FixAgent import FixAgent
import torch

buffer = ReplayBuffer()
epochs = 1500000
learning_rate = 0.01
batch_size = 64
env = Reversi()
iter_number = 1

model = DQN(env)
file='DQN_Model_W_Fix.pth'
model = torch.load(file)
player1 = DQNAgent(model, player=1)
player2 = FixAgent(env, player=2)

def main ():

    # init optimizer
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    state = env.state.toTensor(device=model.device)

    for epoch in range(epochs):
        reward = 0    

        state_Q, _ = player1.get_state_action(state=State.tensorToState(state), epoch=0)

        final_state_s = State.tensorToState(state_Q)
        if env.is_end_of_game(final_state_s):
            state = end_of_game(final_state_s, state_Q, state_Q)
            continue

        next_state, _ = player2.get_state_action(state=State.tensorToState(state_Q), epoch=0)
        final_state_s = State.tensorToState(next_state)
        if env.is_end_of_game(final_state_s):
            state = end_of_game(final_state_s, state_Q, state_Q)
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
            print (f'epoch: {epoch}', end="\r")
        
        if epoch % 10000 == 0:
            torch.save(model, file)        

        if len(buffer) < 500:
            continue

        states_Q, rewards, next_states_Q, dones = buffer.sample(batch_size)
        
        for iter in range(iter_number):
            # zero wights
            optim.zero_grad()

            # forward
            Q_value = model.forward(states_Q)
            with torch.no_grad():
                Q_next_value = model.forward(next_states_Q)

            #backward()
            loss = model.loss(Q_value, rewards, Q_next_value, dones)
            loss.backward()

            if epoch % 1000 == 0 and iter == iter_number-1:
                print (f'epochs: {epoch} loss: {loss}')

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
