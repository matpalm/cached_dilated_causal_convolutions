# by dft we capture two buffers
#  buffer sent to tiliqua; cv_buffer; [ some cvs, v/oct ]
#  buffer received from tiliqua; capture_buffer [ some waves, core triangle ]
# for training we want a dataset that is kind of a join; [phase of triangle, target wave, some cvs ]

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

cap_f = zarr_buffer_fields("capture_buffers.z")
cv_f = zarr_buffer_fields("cv_buffers.z")
md_f = zarr_buffer_fields("inr_model_data.z")

model_data_z = zarr.open(
    zarr_base / "inr_model_data.z",
    mode="w",
    shape=(total_entries, len(md_f)),
    chunks=(chunk_rows, len(md_f)),
    # dtype="<f2",  # f16
    compressor=Blosc(cname="zstd", clevel=4, shuffle=Blosc.SHUFFLE),  # dft clevel=5
)

# have to inline this filter or it's sooooo slow across chunk boundaries
# TODO: am i doing something wrong here?
alpha = 0.6
b = np.array([alpha])
a = np.array([1.0, -(1.0 - alpha)])

zi1 = lfilter_zi(b, a)[:1] * 0.0
zi2 = lfilter_zi(b, a)[:1] * 0.0

tri_amp = 0.54
prev_tri_val = None

for start in tqdm(list(range(0, total_entries, chunk_rows))):
    # read both inputs once per chunk
    end = min(start + chunk_rows, total_entries)
    cap = capture_buffer_z[start:end]
    cv = cv_buffer_z[start:end]

    chunk = np.empty((end - start, len(md_f)), dtype=np.float32)

    cap_tri_out = np.array(cap[:, cap_f.tri_out])

    # convert triangle ( with cycle 0 -> 0.54 -> 0 -> -0.54 -> 0)
    # to a phase as ramp ( with cycle -0.5 -> 0.5 )

    # slope sign disambiguates the two triangle segments for same value
    if prev_tri_val is None:
        # extrapolate back one sample so the first diff is the forward slope.
        first_diff = cap_tri_out[1] - cap_tri_out[0] if len(cap_tri_out) > 1 else 0.0
        prepend_val = cap_tri_out[0] - first_diff
    else:
        prepend_val = prev_tri_val
    prev_tri_val = cap_tri_out[-1]

    diffs = np.diff(cap_tri_out, prepend=prepend_val)
    rising = diffs >= 0
    ramp = np.where(
        rising & (cap_tri_out >= 0),
        cap_tri_out / (2 * tri_amp) - 1,  # rising, positive half: -1.0 -> 0.0
        np.where(
            ~rising,
            -cap_tri_out / (2 * tri_amp),  # falling: -0.5 -> 0.5
            cap_tri_out / (2 * tri_amp) + 1,  # rising, negative half: 0.5 -> 1.0
        ),
    )
    ramp /= 2  # we want +/- 0.5 in training data

    chunk[:, md_f.x_phase] = ramp

    chunk[:, md_f.x_a_cv] = cv[:, cv_f.a_cv]
    chunk[:, md_f.x_b_cv] = cv[:, cv_f.b_cv]
    chunk[:, md_f.x_morph_cv] = cv[:, cv_f.morph_cv]

    # y_true - ( captured ) morph_out ( filtered )
    unfiltered_morph = cap[:, cap_f.morph_out]
    filtered_morph, zi1 = lfilter(b, a, unfiltered_morph, zi=zi1)
    filtered_morph, zi2 = lfilter(b, a, filtered_morph, zi=zi2)
    chunk[:, md_f.y_true] = filtered_morph.astype(np.float32)

    model_data_z[start:end] = chunk  # single write per chunk


print("DONT FORGET TO CONVERT TO NUMPY IF REQUIRED")
