from Reversi import Reversi
from State import State
from DQN import *
from DQNAgent import DQNAgent
from ReplayBuffer import ReplayBuffer
import torch

def main ():

    epochs = 1500000
    learning_rate = 0.1
    batch_size = 64
    env = Reversi()

    model = DQN(env)
    file='DQN_Model.pth'
    # model = torch.load(file)
    player1 = DQNAgent(model, player=1)
    player2 = DQNAgent(model, player=2)

    buffer = ReplayBuffer()

    # init optimizer
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    state = env.state.toTensor(device=model.device)
    state_Q, _ = player1.get_state_action(state=State.tensorToState(state), epoch=0)
    next_state, _ = player2.get_state_action(state=State.tensorToState(state_Q), epoch=0)
    next_state_Q, _ = player1.get_state_action(state=State.tensorToState(next_state), epoch=0)
    final_state, _ = player2.get_state_action(state=State.tensorToState(next_state), epoch=0)

    for epoch in range(epochs):
        reward = 0    
        
        last_state = State.tensorToState(final_state)
        if env.is_end_of_game(last_state):
            reward = last_state.reward(player1.player)
            reward = torch.tensor(reward, dtype=torch.float32, device=model.device)
            # print(player1.player, reward, end="\r")
            # buffer.push(state, -reward, next_state, False)
            buffer.push(state_Q, reward, next_state_Q, True)
            buffer.push(next_state, -reward, next_state, True)
            state = env.get_init_state().toTensor(device=model.device)
            state_Q, _ = player1.get_state_action(state=State.tensorToState(state), epoch=epoch)
            next_state, _ = player2.get_state_action(state=State.tensorToState(state_Q), epoch=epoch)
            next_state_Q, _ = player1.get_state_action(state=State.tensorToState(next_state), epoch=epoch)
            final_state, _ = player2.get_state_action(state=State.tensorToState(next_state), epoch=epoch)
            
            player1.player = 1
            player2.player = 2
            continue

        buffer.push(state_Q, torch.tensor(reward, dtype=torch.float32, device=model.device), next_state_Q, False)
        
        state = state_Q
        state_Q = next_state
        next_state = next_state_Q
        next_state_Q = final_state
        final_state, _ = player1.get_state_action(state=State.tensorToState(next_state), epoch=epoch)        
        player1, player2 = player2, player1

        if epoch % 100 == 0:
            print (epoch, end="\r")
        
        if epoch % 25000 == 0:
            torch.save(model, file)        

        if len(buffer) < 500:
            continue

        states_Q, rewards, next_states_Q, dones = buffer.sample(batch_size)
        
        # zero grads
        optim.zero_grad()

        # forward
        Q_value = model.forward(states_Q)
        with torch.no_grad():
            Q_next_value = model.forward(next_states_Q)
            # Q_next_value.requires_grad_(False)

        #backward()
        loss = model.loss(Q_value, rewards, Q_next_value, dones)
        loss.backward()

        # update wights
        optim.step()

    print (len(buffer))
    
    torch.save(model, file)


if __name__ == '__main__':
    main()
