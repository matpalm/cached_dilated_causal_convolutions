import argparse
from pathlib import Path
import zarr
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from plotting import *
import random


def _fig_to_rgb_array(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return rgba[:, :, :3].copy()


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument(
    "--num",
    type=int,
    default=None,
    help="if set, sample this many randomly for plotting",
)
opts = parser.parse_args()

run_dir = Path("runs") / opts.run

capture_buffers_zarr = zarr.open(run_dir / "capture_buffers.z", mode="r")
cv_buffers_zarr = zarr.open(run_dir / "cv_buffers.z", mode="r")
assert capture_buffers_zarr.shape == cv_buffers_zarr.shape

plots_dir = run_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

idxs = list(range(capture_buffers_zarr.nchunks))
random.seed(123)
random.shuffle(idxs)

# by dft do all, but if --num set, only do that many
num = len(idxs)
if opts.num:
    num = min(num, opts.num)

for i in tqdm(range(num)):
    b_idx = idxs[i]

    cv_buffer = cv_buffers_zarr.blocks[idxs[i]]
    capture_buffer = capture_buffers_zarr.blocks[b_idx]
    assert cv_buffer.shape == capture_buffer.shape

    sample_len, _channels = capture_buffer.shape

    plot_len = 2_000
    plot_offset = random.randint(500, sample_len - plot_len)

    cv_fig = plot(
        cv_buffer,
        title=f"b{b_idx} offset{str(plot_offset)}",
        fname=None,
        plot_offset=plot_offset,
        plot_len=plot_len,
    )
    cv_img_a = _fig_to_rgb_array(cv_fig)

    capture_fig = plot(
        capture_buffer,
        title=f"b{b_idx} offset{str(plot_offset)}",
        fname=None,
        plot_offset=plot_offset,
        plot_len=plot_len,
    )
    capture_img_a = _fig_to_rgb_array(capture_fig)

    combined = np.concatenate([cv_img_a, capture_img_a], axis=1)
    plt.imsave(plots_dir / f"{b_idx:06d}.buffers.jpg", combined)
    plt.close(cv_fig)
    plt.close(capture_fig)
