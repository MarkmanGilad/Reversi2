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
    device = torch.device('cpu')

    model_White = DQN(env)
    model_Black = DQN(env)
    fileWhite='DQN_model_White_P.pth'
    fileBlack='DQN_Model_Black_P.pth'
    # model_White = torch.load(fileWhite)
    # model_Black = torch.load(fileBlack)
    player_White = DQNAgent(model_White, player=1)
    player_Black = DQNAgent(model_Black, player=2)
    player1 = player_White
    player2 = player_Black

    buffer_White = ReplayBuffer()
    buffer_Black = ReplayBuffer()
    buffer1 = buffer_White
    buffer2 = buffer_Black

    # init optimizer
    optim_White = torch.optim.Adam(model_White.parameters(), lr=learning_rate)
    optim_Black = torch.optim.Adam(model_Black.parameters(), lr=learning_rate)
    
    state = env.state.toTensor(device=device)
    state_Q, _ = player1.get_state_action(state=State.tensorToState(state), epoch=0)
    next_state, _ = player2.get_state_action(state=State.tensorToState(state_Q), epoch=0)
    next_state_Q, _ = player1.get_state_action(state=State.tensorToState(next_state), epoch=0)

    for epoch in range(epochs):
        reward = 0    
        
        last_state = State.tensorToState(next_state_Q)
        if env.is_end_of_game(last_state):
            reward = last_state.reward(player1.player)
            reward = torch.tensor(reward, dtype=torch.float32, device=device)
            
            buffer2.push(state, -reward, next_state, False)
            buffer1.push(state_Q, reward, next_state_Q, True)
            buffer2.push(next_state, -reward, next_state, True)
            state = env.get_init_state().toTensor(device=device)
            state_Q, _ = player1.get_state_action(state=State.tensorToState(state), epoch=epoch)
            next_state, _ = player2.get_state_action(state=State.tensorToState(state_Q), epoch=epoch)
            next_state_Q, _ = player1.get_state_action(state=State.tensorToState(next_state), epoch=epoch)
            player1 = player_White
            player2 = player_Black
            buffer1 = buffer_White
            buffer2 = buffer_Black
            continue

        buffer2.push(state, torch.tensor(reward, dtype=torch.float32, device=device), next_state, False)

        player1, player2 = player2, player1
        buffer1, buffer2 = buffer2, buffer1

        state = state_Q
        state_Q = next_state
        next_state = next_state_Q
        next_state_Q, _ = player1.get_state_action(state=State.tensorToState(next_state), epoch=epoch)        
        
        if epoch % 100 == 0:
            print (epoch, end="\r")
        
        if epoch % 25000 == 0:
            torch.save(model_White, fileWhite)
            torch.save(model_Black, fileBlack)

        if len(buffer1) < 500:
            continue

        states_w, rewards_w, next_states_w, dones_w = buffer_White.sample(batch_size)
        states_b, rewards_b, next_states_b, dones_b = buffer_Black.sample(batch_size)
        
        # zero wights
        optim_White.zero_grad()
        optim_Black.zero_grad()

        # forward 
        Q_value_w = model_White.forward(states_w)
        with torch.no_grad():
            Q_next_value_w = model_White.forward(next_states_w)

        Q_value_b = model_Black.forward(states_b)
        with torch.no_grad():
            Q_next_value_b = model_Black.forward(next_states_b)

        #backward()
        loss_w = model_White.loss(Q_value_w, rewards_w, Q_next_value_w, dones_w)
        loss_w.backward()

        loss_b = model_Black.loss(Q_value_b, rewards_b, Q_next_value_b, dones_b)
        loss_b.backward()


        # update wights
        optim_White.step()
        optim_Black.step()

    print (epoch)    
    torch.save(model_White, fileWhite)
    torch.save(model_Black, fileBlack)
    print("saved")


if __name__ == '__main__':
    main()
