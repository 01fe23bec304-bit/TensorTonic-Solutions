import numpy as np

def entropy_node(y):
    if len(y) == 0:
        return 0.0
    
    # Count occurrences of each class
    values, counts = np.unique(y, return_counts=True)
    
    # Compute probabilities
    probs = counts / len(y)
    
    # Remove zero probabilities (for safety)
    probs = probs[probs > 0]
    
    # Compute entropy
    entropy = -np.sum(probs * np.log2(probs))
    
    return float(entropy)