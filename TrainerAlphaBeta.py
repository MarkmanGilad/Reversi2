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
from AlphBetaAgent import AlphaBetaAgent

buffer = ReplayBuffer()
bufferEOG = ReplayBuffer()
epochs = 3000000
learning_rate = 0.01
batch_size = 64
batch_size_EOG = 64
env = Reversi()
testEnv = Reversi()

model = DQN(env)
# model = DQN_tanh(env)
# model = DQN_Sigmoid(env)
file='Reversi/Data/DQN_Model_Fix_eval_LREelu_Point_alphaBeta.pth'
file_best='Reversi/Data/DQN_Model_best_eval_LREelu_Point_alphaBeta.pth'
fileRes = "Reversi/Data/Results_eval_LREelu_Point_alphaBeta.pth"

model = torch.load("Reversi/Data/DQN_Model_Fix_eval_LREelu_Point.pth")
player1 = DQNAgent(model, player=1)
playerFix = FixAgent(env, player=2)
playerRand = RandomAgent(env)
playerFix2 = FixAgent2(env, player=2)
playerAlphBeta = AlphaBetaAgent(player=2, depth=3, environment=env)
player2 = playerAlphBeta

playerTest = DQNAgent(model, player=1, train=False)
testAlphaBeta = tester(env, playerTest, playerAlphBeta)
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
        
        if (epoch) % 100 == 0:
            resFix = testFix.test(1)
            resFix2 = testFix2.test(1)
            resRnd = testRnd.test(100)
            resAlphaBeta = testAlphaBeta.test(1)
            print (f'testFix: {resFix} testFix2: {resFix2} testRand: {resRnd} testAlphaBeta: {resAlphaBeta}')
            
            results.append({'ephocs': epoch+1, 'fix': resFix, 'fix2':resFix2 ,'rnd': resRnd, 'alphaBeta': resAlphaBeta,'loss': loss })
            torch.save(model, file)        
            torch.save(results, fileRes)
            
            if resAlphaBeta[0]*150 + resFix[0]*50 + resFix2[0]* 50 + (resRnd[0]-resRnd[1]) > bestRnd:
                bestRnd = resAlphaBeta[0]*150 + resFix[0]*50 + resFix2[0]* 50 + (resRnd[0]-resRnd[1])
                torch.save(model, file_best) 
                print (f'saveed. epochs: {epoch} loss: {loss} resAlphBeta: {resAlphaBeta} resFix: {resFix} resFix2: {resFix2} resRnd: {resRnd}')
                
        
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
        
        if epoch % 10 == 0:
            print (f'epochs: {epoch}', end="\r")
        
        if len(buffer) < 500:
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
