from amaranth_future import fixed

# model fixed point config
# note: tiliqua codec is FP1.15 but we have model working in FP4.12
N_INT = 4
N_FRAC = 12
NNQ = fixed.SQ(N_INT, N_FRAC)
