import random
import argparse
from pathlib import Path
from tqdm import tqdm
import zarr
import signal
import numpy as np

from .audio_interface import AudioInterface, SAMPLE_RATE_HZ
from .pack_z_array import pack_z_array
from common.sample_db import SampleDB
from .sampling import generate_cv_schroeder_multisine

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run", type=Path, required=True)
# parser.add_argument(
#     "--samples-npy", type=Path, default=None, help="if none, use sampled_db run"
# )
parser.add_argument(
    "--explicitly-use-channels",
    action="store_true",
    help="by dft we 1) add multisines and 2) ch3 is reserved for a v/oct sweep. if this flag is set we use the cv_array values explicitly",
)
parser.add_argument("--sample-len-sec", type=float, default=2.0)
opts = parser.parse_args()

run_dir = Path(__file__).parent / "runs" / opts.run

# fetch from sample_db the count of captures done and pending for this run
db = SampleDB()
captured_stats = db.captured_stats_for(run=opts.run)
print("captured_stats", captured_stats)

# if none are pending, we can exist
if captured_stats[False] == 0:
    print("nothing to do")
    exit()

# if none are done, we need to set things up
if captured_stats[True] == 0:
    run_dir.mkdir(parents=True, exist_ok=True)
    num_todo = captured_stats[False]
    assert num_todo > 0
    sample_len = opts.sample_len_sec * SAMPLE_RATE_HZ
    num_channels = 4
    cv_buffers_z = zarr.open(
        run_dir / "cv_buffers.z",
        mode="w",
        shape=(num_todo * sample_len, num_channels),
        chunks=(sample_len, num_channels),
        dtype=np.float32,
    )
    capture_buffers_z = zarr.open(
        run_dir / "capture_buffers.z",
        mode="w",
        shape=(num_todo * sample_len, num_channels),
        chunks=(sample_len, num_channels),
        dtype=np.float32,
    )
else:
    # we have done some, so just open the existing arrays for append
    cv_buffers_z = zarr.open(run_dir / "cv_buffers.z", mode="a")
    capture_buffers_z = zarr.open(run_dir / "capture_buffers.z", mode="a")

# first check to see if run has been setup; i.e.
run_dir.mkdir(parents=True, exist_ok=True)
# (run_dir / "cv_buffers").mkdir(parents=True, exist_ok=False)
# (run_dir / "capture_buffers").mkdir(parents=True, exist_ok=False)

sample_ids = db.idxs_to_capture(opts.run)
print("pending sample_ids (first 10)", sample_ids[:10])
print("pending sample_ids (last 10)", sample_ids[-10:])

samples = []
for idx in sample_ids:
    samples.append(db.cv_values_for(opts.run, idx))
samples = np.vstack(samples)
print("samples", samples.shape)

amplitudes = samples[:, -1]  # (nb)
cv_values = samples[:, :-1]  # (nb, |cv|)

num_cv_values = cv_values.shape[-1]

if num_cv_values != 3:
    raise Exception(
        "num_cv_values",
        num_cv_values,
        ", can only use 3 ( last reserved for v/oct sweep )",
    )

multisines = generate_cv_schroeder_multisine(
    freq_spacing_hz=5,
    frequencies_per_output=10,
    num_orthogonal=num_cv_values,
    len_sec=opts.sample_len_sec,
)


def cv_a_to_audio_buffer(cv_values, amp):
    assert cv_values.shape == (3,), cv_values.shape
    assert amp >= 0

    num_samples = multisines.shape[-1]
    audio_buffer = np.empty((num_samples, 4), dtype=np.float32)

    if opts.explicitly_use_channels:
        # just explicilty broadcast cv_values across entire sample
        for c_idx, cv_value in enumerate(cv_values):
            audio_buffer[:, c_idx] = cv_value
        audio_buffer[:, 3] = 0
    else:
        # ensure sampled DC + multisine AC bounded
        headroom = np.min(1.0 - np.abs(cv_values))
        effective_amp = min(float(amp), max(0.0, float(headroom)))
        # adjust AC offset of DC cv_values with multisine
        for c_idx, cv_value in enumerate(cv_values):
            audio_buffer[:, c_idx] = cv_value + multisines[c_idx] * effective_amp
        # last output is always voct sweep; 0.0 -> 0.4 -> 0.0 ( ~4octaves, uncalibrated still )
        audio_buffer[:, 3] = np.hstack(
            [
                np.linspace(0.0, 0.4, num_samples // 2),
                np.linspace(0.4, 0.0, num_samples // 2),
            ]
        )

    # ensure in bounds
    np.clip(audio_buffer, -1.0, 1.0, out=audio_buffer)
    return audio_buffer


audio = AudioInterface()

for s, idx in enumerate(tqdm(sample_ids, desc=f"capture {opts.run}")):
    cv_buffer = cv_a_to_audio_buffer(cv_values[s], amplitudes[s])
    cv_buffers_z.blocks[idx] = cv_buffer
    capture_buffer = audio.send(cv_buffer)
    capture_buffers_z.blocks[idx] = capture_buffer
    db.set_captured(run=opts.run, idx=idx)
