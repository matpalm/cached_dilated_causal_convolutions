import numpy as np

from common.util import zarr_buffer_fields
from common.wave_conversion import tri_to_quadrature


def build_triangle_sample(
    seq_len: int,
    a_cv: float,
    b_cv: float,
    morph_cv: float,
    quadrature_input: bool,
    tri_freq: float = 400.0,
    sample_rate: int = 48_000,
):
    """
    Build a synthetic fake x (seq_len, 4)
    """

    f = zarr_buffer_fields("model_data.z")
    n = np.arange(seq_len, dtype=np.float32)
    phase = np.mod(n * (tri_freq / sample_rate), 1.0)
    tri_amp = 0.53  # fixed from capture
    tri = tri_amp * (2.0 * np.abs(2.0 * phase - 1.0) - 1.0)

    if quadrature_input:
        x = np.empty((seq_len, 5), dtype=np.float32)
        sin_q, cos_q = tri_to_quadrature(tri)
        x[:, 0] = sin_q
        x[:, 1] = cos_q
        x[:, 2] = np.float32(a_cv)
        x[:, 3] = np.float32(b_cv)
        x[:, 4] = np.float32(morph_cv)
    else:
        # triangle
        x = np.empty((seq_len, 4), dtype=np.float32)
        x[:, 0] = tri
        x[:, 1] = np.float32(a_cv)
        x[:, 2] = np.float32(b_cv)
        x[:, 3] = np.float32(morph_cv)

    return x
