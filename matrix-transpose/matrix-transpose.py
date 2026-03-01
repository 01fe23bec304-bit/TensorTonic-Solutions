import numpy as np

def matrix_transpose(A):
    A = np.array(A)
    M, N = A.shape
    
    # Create empty matrix with swapped shape
    result = np.zeros((N, M), dtype=A.dtype)
    
    for i in range(M):
        for j in range(N):
            result[j, i] = A[i, j]
    
    return result