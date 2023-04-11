from Reversi import Reversi
from State import State
from DQN import *
from DQNAgent import DQNAgent
from ReplayBuffer import ReplayBuffer
import torch

epochs = 100000
learning_rate = 0.01
batch_size = 64
env = Reversi()


model = DQN(env)
buffer = ReplayBuffer()
# init optimizer
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
file='DQN_Model.pth'



def main ():
    player1 = DQNAgent(model, player=1)
    player2 = DQNAgent(model, player=2)
    
    state = env.state.toTensor(device=model.device)
    state_Q, _ = player1.get_state_action(state=State.tensorToState(state), epoch=0)
    next_state, _ = player2.get_state_action(state=State.tensorToState(state), epoch=0)
    next_state_Q, _ = player1.get_state_action(state=State.tensorToState(state), epoch=0)

    for epoch in range(4, epochs):
        reward = 0    
        
        if env.is_end_of_game(State.tensorToState(next_state_Q)):
            reward = State.tensorToState(next_state).reward(player1.player)
            print(player1.player, reward)
            buffer.push(state_Q, torch.tensor(reward, dtype=torch.float32, device=model.device), next_state_Q, True)
            state = env.get_init_state().toTensor(device=model.device)
            state_Q, _ = player1.get_state_action(state=State.tensorToState(state), epoch=epoch)
            next_state, _ = player2.get_state_action(state=State.tensorToState(state), epoch=epoch)
            next_state_Q, _ = player1.get_state_action(state=State.tensorToState(state), epoch=epoch)
            player1.player = 1
            player2.player = 2
            continue

        buffer.push(state_Q, torch.tensor(reward, dtype=torch.float32, device=model.device), next_state_Q, False)
        
        state = state_Q
        state_Q = next_state
        next_state = next_state_Q
        player1, player2 = player2, player1
        next_state_Q, _ = player1.get_state_action(state=State.tensorToState(state), epoch=epoch)        
        
        if epoch % 100 == 0:
            print (epoch)
        
        if epoch < 500:
            continue

        states, rewards, next_states, dones = buffer.sample(batch_size)
        
        # zero wights
        optim.zero_grad()

        # forward
        Q_value = model.forward(states)
        Q_next_value = model.forward(next_states)

        #backward()
        loss = model.loss(Q_value, rewards, Q_next_value, dones)
        loss.backward()

        # update wights
        optim.step()

    print (len(buffer))
    
    torch.save(model, file)


if __name__ == '__main__':
    main()
