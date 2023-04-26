from Reversi import Reversi
from State import State
from DQN import *
from DQN_tanh import DQN_tanh
from DQN_Sigmoid import DQN_Sigmoid
from DQNAgent import DQNAgent
from ReplayBuffer import ReplayBuffer
from RandomAgent import RandomAgent
from TesterClass import tester
import torch
from AlphBetaAgent import AlphaBetaAgent
from torch.optim import lr_scheduler

buffer = ReplayBuffer()
bufferEOG = ReplayBuffer()
epochs = 5000000
learning_rate = 0.01
batch_size = 64
batch_size_EOG = 64
env = Reversi()
testEnv = Reversi()

model = DQN(env)
file='Reversi/Data/DQN_Model_AB_3.pth'
file_best='Reversi/Data/DQN_Model_best_AB_3.pth'
fileRes = "Reversi/Data/Results_eval_AB_3.pth"

# model = torch.load(file)
player1 = DQNAgent(model, player=1)
player2 = AlphaBetaAgent(player=2, depth=3, environment=env)
playerTest = DQNAgent(model, player=1, train=False)
testFix = tester(env, playerTest, player2)

results = []
# results = torch.load(fileRes)
start = len(results)

def main ():
    bestRnd = 0 
    Games = 0
    loss = 0

    # init optimizer
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optim, step_size=1000000, gamma=0.75)
    # scheduler = lr_scheduler.MultiStepLR(optim, milestones=[500000, 1000000, 1500000, 2000000, 2500000], gamma = 0.7)
    state = env.state.toTensor(device=model.device)
    
    for epoch in range(start, epochs+1):
        
        if (epoch) % 10000 == 0:
            resFix = testFix.test(1)
            print (f'testFix: {resFix}')
            results.append({'ephocs': epoch+1, 'fix': resFix, 'loss': loss })
            torch.save(model, file)        
            torch.save(results, fileRes)
            
            if resFix[0] > bestRnd:
                bestRnd = resFix[0]
                torch.save(model, file_best) 
                print (f'save. epochs: {epoch} loss: {loss}')
        
        reward = 0    

        state_Q, _ = player1.get_state_action(state=State.tensorToState(state), epoch=epoch)

        final_state_s = State.tensorToState(state_Q)
        if env.is_end_of_game(final_state_s):
            state = end_of_game(final_state_s, state_Q, state_Q)
            Games += 1
            continue

        next_state, _ = player2.get_state_action(state=State.tensorToState(state_Q), epoch=epoch)
        final_state_s = State.tensorToState(next_state)
        if env.is_end_of_game(final_state_s):
            state = end_of_game(final_state_s, state_Q, next_state)
            Games += 1
            continue

        next_state_Q, _ = player1.get_state_action(state=State.tensorToState(next_state), epoch=epoch)
        final_state_s = State.tensorToState(next_state_Q)
        if env.is_end_of_game(final_state_s):
            state = end_of_game(final_state_s, state_Q, next_state_Q)
            Games += 1
            continue
    
        final_state, _ = player2.get_state_action(state=State.tensorToState(next_state_Q), epoch=epoch)
        final_state_s = State.tensorToState(final_state)
        if env.is_end_of_game(final_state_s): 
            state = end_of_game(final_state_s, state_Q, next_state_Q)
            Games += 1
            continue
       
        buffer.push(state_Q, torch.tensor(reward, dtype=torch.float32), next_state_Q, False)
      
        state = next_state
        
        print (f'Games: {Games}', end="\r")

        if Games < 5:
            continue   

        states_Q, rewards, next_states_Q, dones = bufferEOG.sample(batch_size_EOG)
        states_Q1, rewards1, next_states_Q1, dones1 = buffer.sample(batch_size_EOG)
        states_Q = buffer.merge_samples(states_Q, states_Q1)
        rewards = buffer.merge_samples(rewards, rewards1)
        next_states_Q = buffer.merge_samples(next_states_Q, next_states_Q1)
        dones = buffer.merge_samples(dones, dones1)

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
            print (f'epoch: {epoch}, loss: {loss}')

        # update wights
        optim.step()
    
    
    torch.save(model, file)

def end_of_game (final_state_s, state_Q, next_state_Q):
    reward = final_state_s.reward(player1.player)
    reward = torch.tensor(reward, dtype=torch.float32)
    bufferEOG.push(state_Q, reward, next_state_Q, True)
    return env.get_init_state().toTensor()

if __name__ == '__main__':
    main()
