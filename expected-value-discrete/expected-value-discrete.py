import numpy as np

def expected_value_discrete(x, p):
    if len(x) != len(p):
        raise ValueError("Length mismatch")
    
    if not np.isclose(np.sum(p), 1):
        raise ValueError("Invalid probabilities")
    
    return float(np.sum(np.array(x) * np.array(p)))