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
assert capture_buffers_zarr.shape == cv_buffers_zarr.shape

("runs" / opts.run / "plots").mkdir(parents=True, exist_ok=True)

idxs = list(range(capture_buffers_zarr.nchunks))
random.seed(123)
random.shuffle(idxs)

for i in tqdm(range(min(len(idxs), opts.num))):
    b_idx = idxs[i]
    capture_buffer = capture_buffers_zarr.blocks[b_idx]
    sample_len, _channels = capture_buffer.shape
    plot_len = 2_000
    plot_offset = random.randint(500, sample_len - plot_len)
    plot(
        capture_buffer,
        title=f"b{b_idx} offset{str(plot_offset)}",
        fname="runs" / opts.run / "plots" / f"{b_idx:06d}.capture_buffer.jpg",
        plot_offset=plot_offset,
        plot_len=plot_len,
    )
    cv_buffer = cv_buffers_zarr.blocks[idxs[i]]
    plot(
        cv_buffer,
        title=f"b{b_idx} offset{str(plot_offset)}",
        fname="runs" / opts.run / "plots" / f"{b_idx:06d}.cv_buffer.jpg",
        plot_offset=plot_offset,
        plot_len=plot_len,
    )
