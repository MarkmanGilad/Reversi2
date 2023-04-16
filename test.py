import numpy as np
import torch
import math
results_W = torch.load('Reversi/Data/Results_AI_AI_W.pth')
results_B = torch.load('Reversi/Data/Results_AI_AI_B.pth')
# print (resulst)
max_W = None
max_B = None
maxScore_W = - 100
maxScore_B = - 100

for res in results_W:
    print (res)
    score = res['fix'][0] *30 + res['fix2'][0]*30 + res['rnd'][0]- res['rnd'][1]
    if  score > maxScore_W:
        maxScore_W = score
        max_W = res

for res in results_B:
    print (res)
    score = res['fix'][1] *30 + res['fix2'][1]*30 + res['rnd'][1]- res['rnd'][0]
    if  score > maxScore_B:
        maxScore_B = score
        max_B = res

print ('max_W', max_W, 'score', maxScore_W)
print ('max_B', max_B, 'score', maxScore_B)



# epsilon Greedy
epsilon_start = 1.0
epsilon_final = 0.05
epsiln_decay = 100000
def epsilon_greedy(epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay):
    res = final + (start - final) * math.exp(-1 * epoch/decay)
    return res

# print (f'{epsilon_greedy(750000):.16f}')