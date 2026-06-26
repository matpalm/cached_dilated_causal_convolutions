import os
from pathlib import Path
import zarr
import numpy as np
import argparse

from plotting import plot

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--zarr", type=Path, required=True)
opts = parser.parse_args()

z = zarr.open(opts.zarr, mode="r")
print("shape", z.shape)
print("chunks", z.chunks)

for i in range(10):
    # sample 1000 steps away from the start
    # just plot first 3 channels
    plot(z.blocks[i][10_000:11_000][:, :3], fname=f"{opts.zarr}_{i}.jpg")
