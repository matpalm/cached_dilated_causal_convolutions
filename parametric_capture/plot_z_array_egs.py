import os
from pathlib import Path
import zarr
import numpy as np
import argparse

from plotting import plot

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--zarr", type=Path, required=True)
parser.add_argument("--idx", type=int, nargs="+", required=False)
parser.add_argument("--plot-offset", type=int, default=10_000)
parser.add_argument("--plot-len", type=int, default=5_000)

opts = parser.parse_args()

z = zarr.open(opts.zarr, mode="r")
print("shape", z.shape)
print("chunks", z.chunks)

r1 = opts.plot_offset
r2 = r1 + opts.plot_len

idxs = opts.idx
if len(opts.idx) == 0:
    idxs = range(10)
for n, idx in enumerate(idxs):
    plot(z.blocks[idx][r1:r2], fname=f"zarr_eg_n{n:03d}_idx{idx:05d}.jpg")
