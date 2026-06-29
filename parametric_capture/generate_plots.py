import argparse
from pathlib import Path
import zarr
from tqdm import tqdm

from plotting import *
import random

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument(
    "--num", type=int, default=16, help="sample this many randomly for plotting"
)
opts = parser.parse_args()

capture_buffers_zarr = zarr.open("runs" / opts.run / "capture_buffers.z", mode="r")
cv_buffers_zarr = zarr.open("runs" / opts.run / "cv_buffers.z", mode="r")

idxs = list(range(capture_buffers_zarr.nchunks))
random.seed(123)
random.shuffle(idxs)
for i in tqdm(range(min(len(idxs), opts.num))):
    b_idx = idxs[i]
    capture_buffer = capture_buffers_zarr.blocks[b_idx]
    plot(
        capture_buffer,
        "runs" / opts.run / "plots" / f"{b_idx:06d}.capture_buffer.jpg",
        plot_offset=10_000,
        plot_len=2_000,
    )
    cv_buffer = cv_buffers_zarr.blocks[idxs[i]]
    plot(
        cv_buffer,
        "runs" / opts.run / "plots" / f"{b_idx:06d}.cv_buffer.jpg",
        plot_offset=10_000,
        plot_len=2_000,
    )
