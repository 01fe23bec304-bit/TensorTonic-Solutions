import numpy as np

def sigmoid(x):
    x = np.array(x, dtype=float)   # ensure numpy array
    return 1 / (1 + np.exp(-x))