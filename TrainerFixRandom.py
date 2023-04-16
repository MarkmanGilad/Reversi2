from Reversi import Reversi
from State import State
from DQN import *
from DQN_tanh import DQN_tanh
from DQN_Sigmoid import DQN_Sigmoid
from DQNAgent import DQNAgent
from ReplayBuffer import ReplayBuffer
from FixAgent import FixAgent
from FixAgent2 import FixAgent2
from RandomAgent import RandomAgent
from TesterClass import tester
import torch

buffer = ReplayBuffer()
epochs = 3000000
learning_rate = 0.01
batch_size = 64
env = Reversi()
testEnv = Reversi()

model = DQN(env)
# model = DQN_tanh(env)
# model = DQN_Sigmoid(env)
file='Data/DQN_Model_Fix_eval_LREelu.pth'
file_best='Data/DQN_Model_best_eval_LREelu.pth'
fileRes = "Data/Results_eval_LREelu.pth"

# model = torch.load("DQN_Model_Fix_eval_tanh.pth")
player1 = DQNAgent(model, player=1)
playerFix = FixAgent(env, player=2)
playerRand = RandomAgent(env)
playerFix2 = FixAgent2(env, player=2)
player2 = playerFix
playerTest = DQNAgent(model, player=1, train=False)
testFix = tester(env, playerTest, playerFix)
testFix2 = tester(env, playerTest, playerFix2)
testRnd = tester(env, playerTest, playerRand)
results = []
# results = torch.load(fileRes)

def main ():
    bestRnd = -100 
    Games = 0
    loss = 0
    # init optimizer
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    state = env.state.toTensor(device=model.device)
    
    for epoch in range(epochs+1):
        
        if (epoch) % 10000 == 0:
            resFix = testFix.test(1)
            resFix2 = testFix2.test(1)
            resRnd = testRnd.test(100)
            print (f'testFix: {resFix}')
            print (f'testFix2: {resFix2}')
            print (f'testRand: {resRnd}')
            results.append({'ephocs': epoch+1, 'fix': resFix, 'fix2':resFix2 ,'rnd': resRnd, 'loss': loss })
            torch.save(model, file)        
            torch.save(results, fileRes)
            
            if resFix[0]*100 + resFix2[0]* 100 + (resRnd[0]-resRnd[1]) > bestRnd:
                bestRnd = resFix[0]*100 + resFix2[0]* 100 + (resRnd[0]-resRnd[1])
                torch.save(model, file_best) 
                print (f'save. epochs: {epoch} loss: {loss}')
                
        
        # if Games % 50 == 0:
        #     if epoch % 10 == 0:
        #         player2 = playerRand
        #     else:
        #         if random.random() < 0.6:
        #             player2 = playerFix
        #         else:
        #             player2 = playerFix2    
        # else:
        #     if random.random() < 0.6:
        #         player2 = playerFix
        #     else:
        #         player2 = playerFix2
        
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
            state = end_of_game(final_state_s, state_Q, next_state_Q)
            Games += 1
            continue

        next_state_Q, _ = player1.get_state_action(state=State.tensorToState(next_state), epoch=epoch)
        final_state_s = State.tensorToState(next_state_Q)
        if env.is_end_of_game(final_state_s):
            state = end_of_game(final_state_s, state_Q, next_state_Q)
            Games += 1
            continue
    
        final_state, _ = player2.get_state_action(state=State.tensorToState(next_state), epoch=epoch)
        final_state_s = State.tensorToState(final_state)
        if env.is_end_of_game(final_state_s): 
            state = end_of_game(final_state_s, state_Q, next_state_Q)
            Games += 1
            continue
       
        buffer.push(state_Q, torch.tensor(reward, dtype=torch.float32), next_state_Q, False)
      
        state = next_state
        
        if epoch % 100 == 0:
            print (f'epochs: {epoch}', end="\r")
        
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
            print (f'epoch: {epoch}, loss: {loss}')

        # update wights
        optim.step()
    
    
    torch.save(model, file)

def end_of_game (final_state_s, state_Q, next_state_Q):
    reward = final_state_s.reward(player1.player)
    reward = torch.tensor(reward, dtype=torch.float32)
    buffer.push(state_Q, reward, next_state_Q, True)
    buffer.push(state_Q, reward, next_state_Q, True)
    buffer.push(state_Q, reward, next_state_Q, True) # triple the end of game data - more accuret and more important
    return env.get_init_state().toTensor()

if __name__ == '__main__':
    main()
