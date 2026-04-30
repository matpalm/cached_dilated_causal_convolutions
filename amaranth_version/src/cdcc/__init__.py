from amaranth_future import fixed

# model fixed point config
# note: tiliqua codec is FP1.15 but we have model working in FP4.12
N_INT = 4
N_FRAC = 12
NNQ = fixed.SQ(N_INT, N_FRAC)

# config for ( double width ) values across dot products and kernel sums
NNQ_DW = fixed.SQ(N_INT * 2, N_FRAC * 2)
