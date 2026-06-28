import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd


class AudioInterface(object):

    NUM_IN_OUT_CHANNELS: int = 4
    BLOCKSIZE = 256  # try 256, 512, or 1024
    LATENCY_SEC = 0.04

    def __init__(self, sample_rate_hz=48_000):

        def scan_for_tiliqua():
            for i, device in enumerate(sd.query_devices()):
                print(i, device)
                if "tiliqua" in device["name"].lower():
                    return i
            raise Exception("not found")

        self.tiliqua_idx = scan_for_tiliqua()
        print("tiliqua_device", sd.query_devices(self.tiliqua_idx)["name"])

        self.sample_rate_hz = sample_rate_hz

    def send(self, buffer):
        recorded_audio = sd.playrec(
            buffer.astype(np.float32),
            samplerate=self.sample_rate_hz,
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


def plot(array, fname=None, plot_offset: int = None, plot_len: int = None):
    if len(array.shape) == 1:
        array = array.reshape((-1, 1))

    if plot_offset is not None:
        array = array[plot_offset : plot_offset + plot_len]

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
    if fname is None:
        plt.close(fig)  # otherwise notebook shows _two_ plots inline :/
        return fig

    fig.savefig(fname, dpi=500)
    plt.close(fig)


SR = 48_000

samples = np.zeros((SR * 4, 4))
up_down = np.hstack(
    [
        np.linspace(-1, 1, SR),
        np.linspace(1, -1, SR),
        np.linspace(-1, 1, SR),
        np.linspace(0, 0, SR),
    ]
)
for i in range(4):
    samples[:, i] = up_down
print("> to_tiliqua")
plot(samples, "test_send_return.to_tiliqua.jpg")

audio = AudioInterface(sample_rate_hz=SR)
recorded_audio = audio.send(samples)

print("< from_tiliqua")
plot(
    recorded_audio,
    "test_send_return.from_tiliqua.jpg",
    plot_offset=10_000,
    plot_len=1_000,
)
