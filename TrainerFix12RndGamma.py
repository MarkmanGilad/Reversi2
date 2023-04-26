from Reversi import Reversi
from State import State
from DQN import *
from DQNAgent import DQNAgent
from ReplayBuffer import ReplayBuffer
from FixAgent import FixAgent
from FixAgent2 import FixAgent2
from TesterClass import tester
import torch
from torch.optim import lr_scheduler


buffer = ReplayBuffer()
bufferEOG = ReplayBuffer()
epochs = 10000000
learning_rate = 0.01
batch_size = 64
batch_size_EOG = 64
env = Reversi()
testEnv = Reversi()

model = DQN(env)
file='Reversi/Data/DQN_Model_Fix_Rnd_Gamma_4.pth'
file_best='Reversi/Data/DQN_Model_best_Fix_Rnd_Gamma_4.pth'
fileRes = "Reversi/Data/Results_eval_Fix_Rnd_Gamma_4.pth"

# model = torch.load(file)
player1 = DQNAgent(model, player=1)
playerFix = FixAgent(env, player=2, train=True)
playerFix2 = FixAgent2(env, player=2, train=True)
player2 = playerFix
player22 = playerFix2
playerTest = DQNAgent(model, player=1, train=False)
testFix = tester(testEnv, playerTest, playerFix)
testFix2 = tester(testEnv, playerTest, playerFix2)

results = []
# results = torch.load(fileRes)
start = len(results)

def main ():
    global player2, player22
    bestRnd = 0 
    Games = 0
    loss = 0

    # init optimizer
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optim, step_size=1000000, gamma=0.75)
    # scheduler = lr_scheduler.MultiStepLR(optim, milestones=[2000000, 4000000, 6000000], gamma = 0.75)
    state = env.state.toTensor(device=model.device)
    
    for epoch in range(start, epochs+1):
        
        if random.random() < 0.5:
            player2, player22 = player22, player2

        if (epoch) % 10000 == 0:
            resFix = testFix.test(100)
            resFix2 = testFix2.test(100)
            print (f'testFix: {resFix}, testFix2: {resFix2}')
            results.append({'ephocs': epoch+1, 'fix': resFix, 'fix2':resFix2,'loss': loss })
            torch.save(model, file)        
            torch.save(results, fileRes)
            
            if (resFix[0] + resFix2[0])/2 > bestRnd:
                bestRnd = (resFix[0] + resFix2[0])/2
                torch.save(model, file_best) 
                print (f'save. epochs: {epoch} bestRnd {bestRnd}, loss: {loss}')
        
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

        #update learning_rate
        scheduler.step()
    
    torch.save(model, file)

def end_of_game (final_state_s, state_Q, next_state_Q):
    reward = final_state_s.reward(player1.player)
    reward = torch.tensor(reward, dtype=torch.float32)
    bufferEOG.push(state_Q, reward, next_state_Q, True)
    return env.get_init_state().toTensor()

if __name__ == '__main__':
    main()
