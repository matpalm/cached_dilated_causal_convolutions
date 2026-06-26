import os
from pathlib import Path
import zarr
import numpy as np

from plotting import plot

z = zarr.open("combined_dataset.zarr", mode="r")
print("shape", z.shape)
print("chunks", z.chunks)

plot(z.blocks[5][1000:2000], fname="z_test_5_340.jpg")
plot(z.blocks[3][4000:5000], fname="z_test_3_340.jpg")
plot(z.blocks[8][7000:8000], fname="z_test_8_340.jpg")
