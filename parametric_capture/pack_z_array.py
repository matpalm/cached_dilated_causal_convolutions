import os
from pathlib import Path
import zarr
import numpy as np
from tqdm import tqdm
import argparse


def pack_z_array(root_npy_dir, output_zarr):
    # count the files and use first to decide shape for all
    fnames = list(sorted(root_npy_dir.iterdir()))
    num_files = len(fnames)
    sample_len, num_channels = np.load(fnames[0]).shape

    # pack into z array
    z = zarr.open(
        output_zarr,
        mode="w",
        shape=(num_files * sample_len, num_channels),
        chunks=(sample_len, num_channels),
        dtype=np.float32,
    )
    for i, fname in enumerate(tqdm(fnames, desc="pack zarr")):
        buffer = np.load(fname)
        assert buffer.shape == (sample_len, num_channels), buffer.shape
        z.blocks[i] = buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--root-npy-dir", type=Path, required=True)
    parser.add_argument("--output-zarr", type=Path, required=True)
    opts = parser.parse_args()
    pack_z_array(opts.root_npy_dir, opts.output_zarr)
