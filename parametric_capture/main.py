import random
import argparse
from pathlib import Path
from tqdm import tqdm

from audio_interface import AudioInterface
from sampling import *
from plotting import *
from util import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run", type=str, required=True)
parser.add_argument(
    "--num-sobol-samples-po2", type=int, default=4, help="should be po2"
)
opts = parser.parse_args()

run_dir = Path("runs") / opts.run
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "plots").mkdir(parents=True, exist_ok=True)
(run_dir / "cv_buffers").mkdir(parents=True, exist_ok=True)
(run_dir / "capture_buffers").mkdir(parents=True, exist_ok=True)

# get initial sobol samples for 3 CV values and A
# note: cv values bounded by (-1, 1) => (-10V, +10V)
#       we use 0.75 instead of 1.0 to avoid very low amp multisines at edges
bounds = []
bounds.append((-0.75, 0.75))  # A cv ; set to noon, (-10, 10)
bounds.append((-0.75, 0.75))  # zero point; full CW,  (-10, 10)
bounds.append((-0.5, 0.5))  # PVM; set to noon,  (-10, 10)
# bounds.append((-1.0, 1.0))  # lin FM  (-10, 10)   ignore for now, not seeming to do anything?
bounds.append((0.2, 1))  # ampltiude
sobol_sampler = SobolSampler(bounds=bounds)
samples = sobol_sampler.samples(num_samples_po2=opts.num_sobol_samples_po2)
print("samples", samples)

np.save(run_dir / "sobol_samples.npy", samples)

# split out amplitude ( last column ) from cv values
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

for s in tqdm(list(range(opts.num_sobol_samples_po2))):

    capture_dts = DTS()

    cv_buffer = cv_a_to_audio_buffer(cv_values[s], amplitudes[s])
    plot(cv_buffer, run_dir / "plots" / f"{capture_dts}.cv_buffer.jpg")
    np.save(run_dir / "cv_buffers" / f"{capture_dts}.npy", cv_buffer)

    capture_buffer = audio.send(cv_buffer)

    plot(
        capture_buffer,
        run_dir / "plots" / f"{capture_dts}.capture_buffer.jpg",
        plot_offset=10_000,
        plot_len=2_000,
    )
    np.save(run_dir / "capture_buffers" / f"{capture_dts}.npy", capture_buffer)

# fade_in_out
# SAMPLE_RATE = 1_000

# sampler = SineWaveSampler(min_freq_hz=0.1, max_freq_hz=5, sample_rate_hz=SAMPLE_RATE)
# samples = []
# for _ in range(4):
#     sample = generate_schroeder_multisine(
#         num_frequencies=5, sample_rate_hz=SAMPLE_RATE, sample_len_s=0.5
#     )
#     # sample = sampler.sample(sample_len_s=0.5)
#     samples.append(sample)
# samples = np.stack(samples).T
# plot(samples, "tiliqua_in.jpg")

# audio = AudioInterface(sample_rate_hz=SAMPLE_RATE)
# recorded_audio = audio.send(samples)
# plot(recorded_audio, "tiliqua_out.jpg")
