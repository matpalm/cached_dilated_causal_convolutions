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


def parse_nnq(v, assert_exact: bool = True, shape=NNQ):
    try:
        iterator = iter(v)
    except TypeError:
        v = float(v)
        fpv = fixed.Const(v, shape=shape)
        if assert_exact and fpv.as_float() != v:
            raise ValueError(
                f"value {v} parsed to NNQ {fpv.as_float()} which isn't exact"
            )
        return fpv
    else:
        return [parse_nnq(v, assert_exact) for v in iterator]
