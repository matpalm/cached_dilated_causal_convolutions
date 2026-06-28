import random
import argparse
from pathlib import Path
from tqdm import tqdm

from audio_interface import AudioInterface
from sampling import *
from plotting import *
from util import *
from pack_z_array import pack_z_array

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument(
    "--samples-npy", type=Path, default=None, help="if none, use run cv_samples.npy"
)
parser.add_argument("--generate-plots", action="store_true")
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
if num_cv_values > 4:
    raise Exception(
        "num_cv_values", num_cv_values, "but there only 4 input channels on device"
    )

multisines = generate_cv_schroeder_multisine(
    freq_spacing_hz=5,
    frequencies_per_output=10,
    num_orthogonal=num_cv_values,
    len_sec=2.0,
)


def cv_a_to_audio_buffer(cv_values, amp):
    num_samples = multisines.shape[-1]
    audio_buffer = np.zeros((num_samples, 4), dtype=np.float32)
    # ensure sampled DC + multisine AC bounded
    headroom = np.min(1.0 - np.abs(cv_values))
    effective_amp = min(float(amp), max(0.0, float(headroom)))
    for c_idx, cv_value in enumerate(cv_values):
        audio_buffer[:, c_idx] = cv_value + multisines[c_idx] * effective_amp
    np.clip(audio_buffer, -1.0, 1.0, out=audio_buffer)
    return audio_buffer


audio = AudioInterface()

for s in tqdm(list(range(len(samples)))):

    capture_dts = DTS()

    cv_buffer = cv_a_to_audio_buffer(cv_values[s], amplitudes[s])

    np.save(run_dir / "cv_buffers" / f"{capture_dts}.npy", cv_buffer)

    if opts.generate_plots:
        plot(cv_buffer, run_dir / "plots" / f"{capture_dts}.cv_buffer.jpg")

    capture_buffer = audio.send(cv_buffer)
    np.save(run_dir / "capture_buffers" / f"{capture_dts}.npy", capture_buffer)

    if opts.generate_plots:
        plot(
            capture_buffer,
            run_dir / "plots" / f"{capture_dts}.capture_buffer.jpg",
            plot_offset=10_000,
            plot_len=2_000,
        )

pack_z_array(run_dir / "capture_buffers", run_dir / "capture_buffers.z")
pack_z_array(run_dir / "cv_buffers", run_dir / "cv_buffers.z")

if str(opts.samples_npy) != str(run_dir / "cv_samples.npy"):
    # we weren't running from the run_dir cv_samples.npy,
    # so save these samples as the definite set now
    np.save(run_dir / "cv_samples.npy", samples)
