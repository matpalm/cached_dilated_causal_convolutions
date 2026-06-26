import os
from pathlib import Path
import zarr
import numpy as np
from tqdm import tqdm

# root_dir = Path("runs/006/capture_buffers/")
root_dir = Path("runs/006/cv_buffers/")

N = 20
SAMPLE_LEN = 96_000
C = 4

z = zarr.open(
    "combined_dataset.zarr",
    mode="w",
    shape=(N * SAMPLE_LEN, C),
    chunks=(SAMPLE_LEN, C),
    dtype=np.float32,
)

fnames = sorted(os.listdir(root_dir))[:N]
for i, fname in enumerate(tqdm(fnames)):
    buffer = np.load(root_dir / fname)
    assert buffer.shape == (SAMPLE_LEN, C), buffer.shape
    z[i * SAMPLE_LEN : (i + 1) * SAMPLE_LEN] = buffer
