from amaranth_future import fixed

from typing import List

# model fixed point config
# note: tiliqua codec is FP1.15 but we have model working in FP4.12
N_INT = 4
N_FRAC = 12
NNQ = fixed.SQ(N_INT, N_FRAC)

# config for ( double width ) values across dot products and kernel sums
NNQ_DW = fixed.SQ(N_INT * 2, N_FRAC * 2)

# all kernels in the network have a kernel size of 4
K = 4

def parse_nnq(values: List[float], check_exact: bool = False):
    fp_values = [fixed.Const(v, shape=NNQ) for v in values]
    if check_exact:
        for fpv, v in zip(fp_values, values):
            if fpv.as_float() != v:
                raise ValueError(
                    f"value {v} parsed to NNQ {fpv.as_float()} which isn't exact"
                )
    return fp_values


# re-export core modules
from .row_by_matrix_multiply import RowByMatrixMultiply
from .conv1d import Conv1d
