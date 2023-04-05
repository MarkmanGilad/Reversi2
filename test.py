import numpy as np
import torch

board = np.array([[1,2], [3,4]])
print (board)
array = board.reshape(-1,)
print(np.append(array, 5))