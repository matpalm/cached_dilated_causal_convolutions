from audio_interface import *
from sampling import *
from plotting import *

SR = 48_000


samples = np.zeros((SR * 2, 4))

# A CV
samples[:, 0] = np.linspace(-0.5, 0.5, SR * 2)  # 0V to +10V
# samples[:, 1] = np.linspace(0.0, 1.0, SR * 2)  # 0V to +10V

audio = AudioInterface(sample_rate_hz=SR)
print(">sending")
recorded_audio = audio.send(samples)
print("<received")
plot(recorded_audio, "test_send_return.from_tiliqua.jpg")
