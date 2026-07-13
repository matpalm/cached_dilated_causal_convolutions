import numpy as np


def build_triangle_sample(
    seq_len,
    a_cv,
    b_cv,
    morph_cv,
    tri_freq=400.0,
    sample_rate=48_000.0,
):
    """
    Build a synthetic fake x (seq_len, 4)
      0 triangle core wave at tri_freq
      1 fixed a_cv
      2 fixed b_cv
      3 fixed morph_cv
    """
    n = np.arange(seq_len, dtype=np.float32)
    phase = np.mod(n * (tri_freq / sample_rate), 1.0)
    tri_amp = 0.53  # fixed from capture
    tri = tri_amp * (2.0 * np.abs(2.0 * phase - 1.0) - 1.0)
    x = np.empty((seq_len, 4), dtype=np.float32)
    x[:, 0] = tri
    x[:, 1] = np.float32(a_cv)
    x[:, 2] = np.float32(b_cv)
    x[:, 3] = np.float32(morph_cv)
    return x
