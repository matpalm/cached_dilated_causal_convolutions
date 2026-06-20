import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# # trying to avoid zeros :/


class AudioInterface(object):

    NUM_IN_OUT_CHANNELS: int = 4
    BLOCKSIZE = 256  # try 256, 512, or 1024
    LATENCY_SEC = 0.04

    def __init__(self, sample_rate=48_000):

        def scan_for_tiliqua():
            for i, device in enumerate(sd.query_devices()):
                if "tiliqua" in device["name"].lower():
                    return i
            raise Exception("not found")

        self.tiliqua_idx = scan_for_tiliqua()
        print("tiliqua_device", sd.query_devices(self.tiliqua_idx)["name"])

        self.sample_rate = sample_rate

    def send(self, buffer):
        recorded_audio = sd.playrec(
            buffer,
            samplerate=self.sample_rate,
            channels=self.NUM_IN_OUT_CHANNELS,
            dtype="float32",
            device=(self.tiliqua_idx, self.tiliqua_idx),
            blocksize=self.BLOCKSIZE,
            latency=(self.LATENCY_SEC, self.LATENCY_SEC),
        )
        sd.wait()
        return recorded_audio

    def num_channels(self):
        return self.NUM_IN_OUT_CHANNELS


def plot(array, fname):
    _, c = array.shape
    fig, axes = plt.subplots(c, 1, figsize=(12, 3 * c), sharex=True)
    axes = np.atleast_1d(axes)

    for i in range(c):
        data = array[:, i]
        axes[i].plot(data)
        axes[i].set_title(f"ch{i}")
        axes[i].set_ylabel("Amplitude")
        axes[i].set_ylim(-1.0, 1.0)

    axes[-1].set_xlabel("sample")
    fig.tight_layout()
    fig.savefig(fname, dpi=50)
    plt.close(fig)


# CV out; 0s, then 0.5,
SAMPLE_RATE = 48000
SAMPLE_LEN_SEC = 5.0  # Record/playback duration in seconds
NUM_SAMPLES = int(SAMPLE_RATE * SAMPLE_LEN_SEC)
NUM_IN_OUT_CHANNELS = 4

audio = AudioInterface(sample_rate=SAMPLE_RATE)

# make a sine wave for tiliqua to send  ( +/- => +/-10V )
t = np.arange(NUM_SAMPLES, dtype=np.float32) / NUM_SAMPLES
cv_out = np.sin(2.0 * np.pi * 3.0 * t)

# pack it into the buffer ( sd will send this info )
buffer = np.zeros((NUM_SAMPLES, audio.num_channels()), dtype=np.float32)
buffer[:, 0] = cv_out
buffer[:, 1] = -cv_out
buffer[:, 2] = cv_out * 0.5

plot(buffer, "out.jpg")

recorded_audio = audio.send(buffer)

plot(recorded_audio, "ch_in.jpg")
