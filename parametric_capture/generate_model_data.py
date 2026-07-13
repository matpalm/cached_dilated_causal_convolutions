# by dft we capture two buffers
#  buffer sent to tiliqua; cv_buffer; [ some cvs, v/oct ]
#  buffer received from tiliqua; capture_buffer [ some waves, core triangle ]
# for training we want a dataset that is kind of a join; [core_triangle, target wave, some cvs ]

from pathlib import Path
import zarr
import numpy as np
from scipy.signal import lfilter, lfilter_zi
import argparse
from tqdm import tqdm
from numcodecs import Blosc

from common.util import zarr_base_path_for, zarr_buffer_fields

# def two_stage_one_pole_lowpass(x: np.ndarray, alpha: float = 0.6) -> np.ndarray:
#     b = np.array([alpha])
#     a = np.array([1.0, -(1.0 - alpha)])
#     stage1 = lfilter(b, a, x)
#     stage2 = lfilter(b, a, stage1)
#     return stage2

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run", type=Path, required=True)
opts = parser.parse_args()

zarr_base = zarr_base_path_for(opts.run)

capture_buffer_z = zarr.open(zarr_base / "capture_buffers.z", mode="r")
cv_buffer_z = zarr.open(zarr_base / "cv_buffers.z", mode="r")
assert capture_buffer_z.nchunks == cv_buffer_z.nchunks
print("capture_buffer_z.nchunks", capture_buffer_z.nchunks)
assert capture_buffer_z.shape == cv_buffer_z.shape
print("capture_buffer_z.shape", capture_buffer_z.shape)
total_entries = capture_buffer_z.shape[0]

chunk_rows = capture_buffer_z.chunks[0]

model_data_z = zarr.open(
    zarr_base / "model_data.z",
    mode="w",
    shape=(total_entries, 5),
    chunks=(chunk_rows, 5),
    dtype="<f2",  # f16
    compressor=Blosc(cname="zstd", clevel=4, shuffle=Blosc.SHUFFLE),  # dft clevel=5
)

# have to inline this filter or it's sooooo slow across chunk boundaries
# TODO: am i doing something wrong here?
alpha = 0.6
b = np.array([alpha])
a = np.array([1.0, -(1.0 - alpha)])

zi1 = lfilter_zi(b, a)[:1] * 0.0
zi2 = lfilter_zi(b, a)[:1] * 0.0

cap_f = zarr_buffer_fields("capture_buffers.z")
cv_f = zarr_buffer_fields("cv_buffers.z")
md_f = zarr_buffer_fields("model_data.z")

for start in tqdm(list(range(0, total_entries, chunk_rows))):
    # read both inputs once per chunk
    end = min(start + chunk_rows, total_entries)
    cap = capture_buffer_z[start:end]
    cv = cv_buffer_z[start:end]

    C = 5
    chunk = np.empty((end - start, C), dtype=np.float32)
    chunk[:, md_f.x_tri] = cap[:, cap_f.tri_out]
    chunk[:, md_f.x_a_cv] = cv[:, cv_f.a_cv]
    chunk[:, md_f.x_b_cv] = cv[:, cv_f.b_cv]
    chunk[:, md_f.morph_cv] = cv[:, cv_f.morph_cv]

    # y_true - ( captured ) morph_out ( filtered )
    unfiltered_morph = cap[:, cap_f.morph_out]
    filtered_morph, zi1 = lfilter(b, a, unfiltered_morph, zi=zi1)
    filtered_morph, zi2 = lfilter(b, a, filtered_morph, zi=zi2)
    chunk[:, md_f.y_true] = filtered_morph.astype(np.float32)

    model_data_z[start:end] = chunk  # single write per chunk
