import argparse
from pathlib import Path
import zarr
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

from .plotting import collage, plot
from common.util import zarr_base_path_for

def _fig_to_pil_img(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return Image.fromarray(rgba[:, :, :3].copy())


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument(
    "--num",
    type=int,
    default=None,
    help="if set, sample this many randomly for plotting",
)
opts = parser.parse_args()

run_dir = zarr_base_path_for(opts.run)

capture_buffers_zarr = zarr.open(run_dir / "capture_buffers.z", mode="r")
cv_buffers_zarr = zarr.open(run_dir / "cv_buffers.z", mode="r")
model_data_zarr = zarr.open(run_dir / "model_data.z", mode="r")
assert capture_buffers_zarr.shape == cv_buffers_zarr.shape
assert len(capture_buffers_zarr) == len(model_data_zarr)

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
    cv_buffer = cv_buffers_zarr.blocks[b_idx]
    capture_buffer = capture_buffers_zarr.blocks[b_idx]
    model_data_buffer = model_data_zarr.blocks[b_idx]
    plots = [plot(cv_buffer), plot(capture_buffer), plot(model_data_buffer)]
    combined = collage(plots, side_by_side=True)
    combined.save(plots_dir / f"{b_idx:06d}.buffers.jpg")
