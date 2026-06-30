import random
import argparse
from pathlib import Path
from tqdm import tqdm

from audio_interface import AudioInterface
from sampling import *
from util import *
from pack_z_array import pack_z_array

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument(
    "--samples-npy", type=Path, default=None, help="if none, use run cv_samples.npy"
)
parser.add_argument(
    "--explicitly-use-channels",
    action="store_true",
    help="by dft we 1) add multisines and 2) ch3 is reserved for a v/oct sweep. if this flag is set we use the cv_array values explicitly",
)
parser.add_argument("--sample-len-sec", type=float, default=2.0)
opts = parser.parse_args()

run_dir = Path("runs") / opts.run
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "plots").mkdir(parents=True, exist_ok=False)
(run_dir / "cv_buffers").mkdir(parents=True, exist_ok=False)
(run_dir / "capture_buffers").mkdir(parents=True, exist_ok=False)

samples_npy = opts.samples_npy
if samples_npy is None:
    samples_npy = Path("runs") / opts.run / "cv_samples.npy"

samples = np.load(samples_npy)
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

for s in tqdm(list(range(len(samples))), desc="capture"):
    capture_dts = DTS()
    cv_buffer = cv_a_to_audio_buffer(cv_values[s], amplitudes[s])
    np.save(run_dir / "cv_buffers" / f"{capture_dts}.npy", cv_buffer)
    capture_buffer = audio.send(cv_buffer)
    np.save(run_dir / "capture_buffers" / f"{capture_dts}.npy", capture_buffer)

pack_z_array(run_dir / "capture_buffers", run_dir / "capture_buffers.z")
pack_z_array(run_dir / "cv_buffers", run_dir / "cv_buffers.z")

if str(opts.samples_npy) != str(run_dir / "cv_samples.npy"):
    # we weren't running from the run_dir cv_samples.npy,
    # so save these samples as the definite set now
    np.save(run_dir / "cv_samples.npy", samples)
