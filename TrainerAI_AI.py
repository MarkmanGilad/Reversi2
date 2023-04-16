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

buffer_W = ReplayBuffer()
buffer_W_EOG = ReplayBuffer()
buffer_B = ReplayBuffer()
buffer_B_EOG = ReplayBuffer()
epochs = 3000000
learning_rate = 0.01
batch_size = 64
batch_size_EOG = 64
env = Reversi()
testEnv = Reversi()
device = 'cpu'

model_W = DQN(env)
model_B = DQN(env)


file_W='Data/DQN_Model_AI_AI_W.pth'
file_best_W='Data/DQN_Model_AI_AI_best_eval_W.pth'
fileRes_W = "Data/Results_AI_AI_W.pth"
file_B='Data/DQN_Model_AI_AI_B.pth'
file_best_B='Data/DQN_Model_AI_AI_best_eval_B.pth'
fileRes_B = "Data/Results_AI_AI_B.pth"

# model_W = torch.load("Data/DQN_Model_AI_AI_W.pth")
# model_B = torch.load("Data/DQN_Model_AI_AI_B.pth")

player_W = DQNAgent(model_W, player=1)
player_B = DQNAgent(model_B, player=2)

playerFix_W = FixAgent(env, player=1)
playerFix_B = FixAgent(env, player=2)
playerFix2_W = FixAgent2(env, player=1)
playerFix2_B = FixAgent2(env, player=2)
playerRand = RandomAgent(env)

playerTest_W = DQNAgent(model_W, player=1, train=False)
playerTest_B = DQNAgent(model_B, player=2, train=False)

testFix_W = tester(env, playerTest_W, playerFix_B)
testFix2_W = tester(env, playerTest_W, playerFix2_B)
testFix_B = tester(env, playerFix_W, playerTest_B)
testFix2_B = tester(env,playerFix_W, playerTest_B)
testRnd_W = tester(env, playerTest_W, playerRand)
testRnd_B = tester(env, playerRand, playerTest_B)
results_W = []
results_B = []
# results = torch.load(fileRes)

bestScore_W = -100
bestScore_B = -100

def main ():
    best_W = -100 
    best_B = -100 

    Games = 0
    loss = 0

    # init optimizer
    optim_W = torch.optim.Adam(model_W.parameters(), lr=learning_rate)
    optim_B = torch.optim.Adam(model_B.parameters(), lr=learning_rate)
    optim2 = optim_W
    optim1 = optim_B

    # players    
    player2 = player_W
    player1 = player_B
    buffer2 = buffer_W
    buffer1 = buffer_B
    buffer2_EOG = buffer_W_EOG
    buffer1_EOG = buffer_B_EOG
    
    model2 = model_W
    model1 = model_B
    
    epoch = 0
    state = env.state.toTensor(device=device)
    state_Q, _ = player2.get_state_action(state=State.tensorToState(state), epoch=epoch)
    next_state, _ = player1.get_state_action(state=State.tensorToState(state_Q), epoch=epoch)
    next_state_Q, _ = player2.get_state_action(state=State.tensorToState(next_state), epoch=epoch)
    
    for epoch in range(epochs+1):
        
        if (epoch) % 10000 == 0:
            saveModels(epoch, loss)
            
        player1, player2 = player2, player1
        optim1, optim2 = optim2, optim1
        buffer1, buffer2 = buffer2, buffer1
        buffer1_EOG, buffer2_EOG = buffer2_EOG, buffer1_EOG
        model1, model2 = model2, model1

        reward = 0    
    
        final_state, _ = player2.get_state_action(state=State.tensorToState(next_state), epoch=epoch)
        final_state_s = State.tensorToState(final_state)
        if env.is_end_of_game(final_state_s): 
            end_of_game(buffer1_EOG, player1, final_state_s, state_Q, next_state_Q)
            end_of_game(buffer2_EOG, player2, final_state_s, next_state, final_state)
            state = env.get_init_state().toTensor()
            state_Q, _ = player1.get_state_action(state=State.tensorToState(state), epoch=epoch)
            next_state, _ = player2.get_state_action(state=State.tensorToState(state_Q), epoch=epoch)
            next_state_Q, _ = player1.get_state_action(state=State.tensorToState(next_state), epoch=epoch)
            Games += 1
            player2 = player_W
            player1 = player_B
            buffer2 = buffer_W
            buffer1 = buffer_B
            buffer2_EOG = buffer_W_EOG
            buffer1_EOG = buffer_B_EOG
            model2 = model_W
            model1 = model_B    
            optim2 = optim_W
            optim1 = optim_B
            continue
       
        buffer1.push(state_Q, torch.tensor(reward, dtype=torch.float32), next_state_Q, False)
      
        state = state_Q
        state_Q = next_state
        next_state = next_state_Q
        next_state_Q = final_state
        
        
        print (f'epochs: {epoch}', end="\r")
        
        if Games < 5:
            continue

        states_Q, rewards, next_states_Q, dones = buffer1_EOG.sample(batch_size_EOG)
        states_Q1, rewards1, next_states_Q1, dones1 = buffer1.sample(batch_size_EOG)
        states_Q = buffer1.merge_samples(states_Q, states_Q1)
        rewards = buffer1.merge_samples(rewards, rewards1)
        next_states_Q = buffer1.merge_samples(next_states_Q, next_states_Q1)
        dones = buffer1.merge_samples(dones, dones1)

        # zero wights
        optim1.zero_grad()

        # forward
        Q_value = model1.forward(states_Q)
        with torch.no_grad():
            Q_next_value = model1.forward(next_states_Q)

        #backward()
        loss = model1.loss(Q_value, rewards, Q_next_value, dones)
        loss.backward()

        if epoch % 1000 == 0:
            print (f'epoch: {epoch}, loss: {loss}')

        # update wights
        optim1.step()
    

def end_of_game (buffer, player, final_state_s, state_Q, next_state_Q):
    reward = final_state_s.reward(player.player)
    reward = torch.tensor(reward, dtype=torch.float32)
    buffer.push(state_Q, reward, next_state_Q, True)
    
def saveModels (epoch, loss):
    global bestScore_W, bestScore_B
    
    resFix_W = testFix_W.test(1)
    resFix2_W = testFix2_W.test(1)
    resRnd_W = testRnd_W.test(100)
    print (f'white - testFix: {resFix_W} testFix2: {resFix2_W} testRand: {resRnd_W}')
    results_W.append({'ephocs': epoch, 'fix': resFix_W, 'fix2':resFix2_W ,'rnd': resRnd_W, 'loss': loss })
    torch.save(model_W, file_W)        
    torch.save(results_W, fileRes_W)

    resFix_B = testFix_B.test(1)
    resFix2_B = testFix2_B.test(1)
    resRnd_B = testRnd_B.test(100)
    print (f'Black - testFix: {resFix_B} testFix2: {resFix2_B} testRand: {resRnd_B}')
    results_B.append({'ephocs': epoch, 'fix': resFix_B, 'fix2':resFix2_B ,'rnd': resRnd_B, 'loss': loss })
    torch.save(model_B, file_B)        
    torch.save(results_B, fileRes_B)
        
    if resFix_W[0]*30 + resFix2_W[0]* 30 + (resRnd_W[0]-resRnd_W[1]) > bestScore_W :
         bestScore_W = resFix_W[0]*30 + resFix2_W[0]* 30 + (resRnd_W[0]-resRnd_W[1])
         torch.save(model_W, file_best_W) 
         print (f'save W. epochs: {epoch} best score: {bestScore_W}')
    
    if resFix_B[1]*30 + resFix2_B[1]* 30 + (resRnd_B[1]-resRnd_B[0]) > bestScore_B :
         bestScore_B = resFix_B[1]*30 + resFix2_B[1]* 30 + (resRnd_B[1]-resRnd_B[0])
         torch.save(model_B, file_best_B) 
         print (f'save B. epochs: {epoch} best score: {bestScore_B}')

if __name__ == '__main__':
    main()
