from audio_interface import *
from sampling import *
from plotting import *

SR = 48_000

sampler = SineWaveSampler(min_freq_hz=100, max_freq_hz=120, sample_rate_hz=SR)
samples = []
for _ in range(4):
    samples.append(sampler.sample(sample_len_s=2.0))

samples = np.stack(samples).T
plot(samples, "test_send_return.to_tiliqua.jpg", plot_offset=10_000, plot_len=2_000)

audio = AudioInterface(sample_rate_hz=SR)

print("samples shape:", samples.shape)
print("samples dtype:", samples.dtype)
print("samples max abs:", np.max(np.abs(samples)))

import sounddevice as sd

print(sd.query_devices(audio.tiliqua_idx))

recorded_audio = audio.send(samples)
print("<received")
plot(
    recorded_audio,
    "test_send_return.from_tiliqua.jpg",
    plot_offset=10_000,
    plot_len=2_000,
)
