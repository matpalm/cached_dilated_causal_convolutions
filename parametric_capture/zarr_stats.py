import zarr
import argparse
from pathlib import Path
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from .util import zarr_to_columns

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--zarr-dir", type=Path, required=True)
parser.add_argument(
    "--max-blocks",
    type=int,
    default=50,
    help="max number of shuffled blocks to sample",
)
opts = parser.parse_args()

z = zarr.open(opts.zarr_dir, "r")
print(z.shape)
column_names = zarr_to_columns(opts.zarr_dir.name)
num_cols = len(column_names)
values_by_col = [[] for _ in range(num_cols)]
blocks = list(range(z.nchunks))
random.seed(1234)
random.shuffle(blocks)
for i, b in enumerate(blocks):
    trim = 1000
    block = z.blocks[b][trim:-trim]

    if block.shape[-1] != num_cols:
        raise ValueError(
            f"mismatch in col names? |cols|={num_cols} but block {block.shape[-1]}"
        )

    for col_idx in range(num_cols):
        column = block[..., col_idx]
        every_5th = np.ravel(column[::5])
        values_by_col[col_idx].append(every_5th)

    if i + 1 >= opts.max_blocks:
        break

concatenated = [np.concatenate(v).astype(np.float32) for v in values_by_col]
for name, values in zip(column_names, concatenated):
    deciles = np.percentile(values, np.linspace(0, 100, 11))
    print(f"{name:>10}  values={values.shape}  deciles={list(np.around(deciles, 3))}")

# plot all distributions overlaid
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    for name, values in zip(column_names, concatenated):
        sns.histplot(
            values,
            bins=120,
            stat="density",
            kde=True,
            alpha=0.25,
            element="step",
            fill=True,
            label=name,
            ax=ax,
        )

ax.set_xlim(-1, 1)
ax.set_xlabel("value")
ax.set_ylabel("density")
ax.set_title(f"Overlaid distributions: {opts.zarr_dir.name}")
ax.legend(loc="best")
fig.tight_layout()
out_path = opts.zarr_dir.parent / f"{opts.zarr_dir.stem}_dists.png"
fig.savefig(out_path)
plt.close(fig)
print("saved plot", out_path)
