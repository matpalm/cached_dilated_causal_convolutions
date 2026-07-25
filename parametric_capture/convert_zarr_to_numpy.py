import argparse
from pathlib import Path
import zarr
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--src-zarr", type=Path, required=True, help="path to input zarr array"
)
parser.add_argument(
    "--output-npy", type=Path, required=True, help="path to output .npy file"
)
parser.add_argument("--output-dtype", type=str, default="float32")

opts = parser.parse_args()

z = zarr.open(str(opts.src_zarr), mode="r")
print("opened", opts.src_zarr, "shape", z.shape, "dtype", z.dtype, "nchunks", z.nchunks)
assert z.ndim == 2, f"expected (N, F) zarr, got shape {z.shape}"
seq_len, num_features = z.chunks

np_shape = (z.nchunks, seq_len, num_features)
np_dtype = np.dtype(opts.output_dtype)
print("writing npy shape", np_shape, "np_dtype", np_dtype)
arr = np.lib.format.open_memmap(
    opts.output_npy,
    mode="w+",
    dtype=np_dtype,
    shape=np_shape,
)

for i in tqdm(range(z.nchunks), desc="write blocks"):
    block = z.blocks[i]
    assert block.shape == (seq_len, num_features), block.shape
    arr[i] = block

arr.flush()
print("wrote", opts.output_npy, "shape", arr.shape)
