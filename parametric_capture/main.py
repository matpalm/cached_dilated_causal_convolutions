import random

from audio_interface import AudioInterface
from sampling import *
from plotting import *
from util import *

SR = 48_000

# get initial sobol samples for 3 CV values and A
# note: cv values bounded by (-1, 1) => (-10V, +10V)
num_sobol_samples = 4
sobol_sampler = SobolSampler(bounds=[(0, 0.5), (-0.5, 0.5), (0.01, 1)])
samples = sobol_sampler.samples(num_samples_po2=num_sobol_samples)
print("samples", samples)

# split out amplitude ( last column ) from cv values
amplitudes = samples[:, -1]  # (nb)
cv_values = samples[:, :-1]  # (nb, |cv|)
print("amplitude", amplitudes)
print("cv_values", cv_values)

num_cv_values = cv_values.shape[-1]
print("num_cv_values", num_cv_values)
if num_cv_values > 4:
    raise Exception(
        "num_cv_values", num_cv_values, "but there only 4 input channels on device"
    )


def cv_a_to_audio_buffer(cv_values, amp):
    multisines = generate_cv_schroeder_multisine(
        freq_spacing_hz=5,
        frequencies_per_output=10,
        num_orthogonal=len(cv_values),
        sample_rate_hz=SR,
        len_sec=2.0,
    )
    num_samples = multisines.shape[-1]
    audio_buffer = np.zeros((num_samples, 4), dtype=float)
    for c_idx, cv_value in enumerate(cv_values):
        audio_buffer[:, c_idx] = cv_value + multisines[c_idx] * amp
    return audio_buffer


for s in range(num_sobol_samples):
    print("cv_values", cv_values[s])
    audio_buffer = cv_a_to_audio_buffer(cv_values[s], amplitudes[s])
    audio_buffer = fade_in_out(audio_buffer, fade_num_samples=500)
    plot(audio_buffer, f"eg_audio_buffer.{s}.jpg")

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
