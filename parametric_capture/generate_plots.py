import argparse
from pathlib import Path
import zarr
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from plotting import collage, plot
import random


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

run_dir = Path("runs") / opts.run

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

    sample_len, _channels = capture_buffer.shape

    plot_len = 2_000
    plot_offset = random.randint(500, sample_len - plot_len)

    def plot_buffer(buffer):
        fig = plot(
            buffer,
            title=f"b{b_idx} offset{str(plot_offset)}",
            fname=None,
            plot_offset=plot_offset,
            plot_len=plot_len,
        )
        pil_img = _fig_to_pil_img(fig)
        plt.close(fig)
        return pil_img

    cv_pil_img = plot_buffer(cv_buffer)
    capture_pil_img = plot_buffer(capture_buffer)
    model_pil_img = plot_buffer(model_data_buffer)

    combined = collage([cv_pil_img, capture_pil_img, model_pil_img], side_by_side=True)
    combined.save(plots_dir / f"{b_idx:06d}.buffers.jpg")
