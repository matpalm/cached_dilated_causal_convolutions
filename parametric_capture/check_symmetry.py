import argparse
from pathlib import Path
import numpy as np
import zarr
import pandas as pd

from common.util import zarr_base_path_for
from common.sample_db import SampleDB
from common.losses import combined_masked_loss_terms
from parametric_capture.plotting import plot

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run", type=Path, required=True)
opts = parser.parse_args()

db = SampleDB()

zarr_base = zarr_base_path_for(opts.run)
capture_buffer_z = zarr.open(zarr_base / "capture_buffers.z", mode="r")

# cv_buffer_z = zarr.open(zarr_base / "cv_buffers.z", mode="r")
cv_values = db.cv_values_for(opts.run)[:, :3]

# df = pd.DataFrame(cv_values, columns=["a_cv", "morph", "b_cv"])
# print(df.describe())


def nearest_idx_for(idx: int, flipped: bool):
    """
    find the nearest neighbour for row idx _except_ with
    a_cv and b_cv swapped and m negated
    """
    if flipped:
        a_cv, m, b_cv = cv_values[idx]
        cv_value = np.array([b_cv, -m, a_cv])
    else:
        cv_value = cv_values[idx]
    # brute force is fine
    deltas = cv_values - cv_value
    distances = np.sqrt(np.sum(deltas**2, axis=1))
    distances[idx] = np.inf
    return int(np.argmin(distances))


idx_i = 234
print("i ", cv_values[idx_i])
idx_j = nearest_idx_for(idx_i, flipped=False)
print("j ", cv_values[idx_j])
idx_jf = nearest_idx_for(idx_i, flipped=True)
print("jf", cv_values[idx_jf])


def captured_audio(capture_buffer_block):
    # return the capture waveshape as (1, S, 1) ready for loss call
    return capture_buffer_block[:, 0][np.newaxis, :, np.newaxis]


sample_i = captured_audio(capture_buffer_z.blocks[idx_i])
sample_j = captured_audio(capture_buffer_z.blocks[idx_j])
sample_jf = captured_audio(capture_buffer_z.blocks[idx_jf])

plot(sample_i[0], plot_offset=10_000, plot_len=2_000).save("sample_i.png")
plot(sample_j[0], plot_offset=10_000, plot_len=2_000).save("sample_j.png")
plot(sample_jf[0], plot_offset=10_000, plot_len=2_000).save("sample_jf.png")


loss_fn, huber_loss_fn, stft_loss_fn = combined_masked_loss_terms(
    receptive_field_size=None,
    use_huber_loss=True,
    alpha_mse=1.0,
    beta_stft=0.01,
    reduce_mean=True,
)
print("i j", stft_loss_fn(sample_i, sample_j))
print("i jf", stft_loss_fn(sample_i, sample_jf))
print("j jf", stft_loss_fn(sample_j, sample_jf))
