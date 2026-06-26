import matplotlib.pyplot as plt
import numpy as np


def plot(array, fname, plot_offset: int = None, plot_len: int = None):
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
    fig.savefig(fname, dpi=500)
    plt.close(fig)


def plot_spectrogram(series, fname, sample_rate_hz, max_freq_hz: float = 6000):
    # NFFT=4096 gives ~2.9 Hz bins at 12 kHz, resolving 5 Hz spacing
    nfft = 4096
    noverlap = nfft * 3 // 4
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.specgram(series, Fs=sample_rate_hz, NFFT=nfft, noverlap=noverlap, cmap="inferno")
    ax.set_ylim(0, max_freq_hz)
    ax.set_yticks(np.arange(0, max_freq_hz + 1, 5))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("spectrogram")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
