import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0):
    # positions: (seq_len, 1)
    pos = np.arange(seq_len)[:, np.newaxis]

    # dimension indices: (1, d_model)
    i = np.arange(d_model)[np.newaxis, :]

    # angle rates
    angle_rates = pos / np.power(base, (2 * (i // 2)) / d_model)

    # apply sin to even indices, cos to odd indices
    pe = np.zeros_like(angle_rates)
    pe[:, 0::2] = np.sin(angle_rates[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rates[:, 1::2])

    return pe