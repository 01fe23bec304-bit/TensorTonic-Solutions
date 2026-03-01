import numpy as np

def rmsprop_step(w, g, s, lr=0.01, beta=0.9, eps=1e-8):
    # Convert inputs to NumPy arrays (important)
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    s = np.array(s, dtype=float)

    # Update running squared gradient
    s_new = beta * s + (1 - beta) * (g ** 2)

    # Parameter update
    w_new = w - lr * g / (np.sqrt(s_new) + eps)

    return w_new, s_new