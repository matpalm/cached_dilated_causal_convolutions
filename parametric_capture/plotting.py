# seaborn just wont shut up
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from audio_interface import SAMPLE_RATE_HZ

def plot(
    array, title: str = "", fname=None, plot_offset: int = None, plot_len: int = None
):
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
        axes[i].set_title(f"ch{i} {title}")
        axes[i].set_ylabel("Amplitude")
        axes[i].set_ylim(-1.0, 1.0)

    axes[-1].set_xlabel("sample")
    fig.tight_layout()
    if fname is None:
        plt.close(fig)  # otherwise notebook shows _two_ plots inline :/
        return fig

    fig.savefig(fname, dpi=500)
    plt.close(fig)


def plot_spectrogram(series, fname, max_freq_hz: float = 6_000):
    # NFFT=4096 gives ~2.9 Hz bins at 12 kHz, resolving 5 Hz spacing
    nfft = 4096
    noverlap = nfft * 3 // 4
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.specgram(series, Fs=SAMPLE_RATE_HZ, NFFT=nfft, noverlap=noverlap, cmap="inferno")
    ax.set_ylim(0, max_freq_hz)
    ax.set_yticks(np.arange(0, max_freq_hz + 1, 5))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("spectrogram")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def play_sample(
    series,
    play_offset: int = None,
    play_len: int = None,
    autoplay: bool = True,
):
    from IPython.display import Audio

    series = np.asarray(series)
    if play_offset is not None:
        series = series[play_offset : play_offset + play_len]
    return Audio(series.astype(np.float32), rate=SAMPLE_RATE_HZ, autoplay=autoplay)


def collage(pil_imgs, side_by_side: bool = False, stacked: bool = False):
    if not (side_by_side ^ stacked):
        raise Exception("side_by_side xor stacked")
    # if passed something that isn't pil images, try to convert...
    # if type(pil_imgs[0]) != Image:
    #    pil_imgs = [img_a_to_pil(a) for a in pil_imgs]
    # use largest w and h for collage cells
    w, h = 0, 0
    for p in pil_imgs:
        iw, ih = p.size
        w = max(w, iw)
        h = max(h, ih)
    w, h = w + 2, h + 2
    if side_by_side:
        collage = Image.new("RGB", (len(pil_imgs) * w, h), (255, 0, 0))
        for idx, img in enumerate(pil_imgs):
            collage.paste(img, (idx * w, 0))
    elif stacked:
        collage = Image.new("RGB", (w, len(pil_imgs) * h), (255, 0, 0))
        for idx, img in enumerate(pil_imgs):
            collage.paste(img, (0, idx * h))
    else:
        n = math.ceil(math.sqrt(len(pil_imgs)))
        collage = Image.new("RGB", (n * w, n * h), (255, 0, 0))
        for idx, img in enumerate(pil_imgs):
            r, c = idx % n, idx // n
            collage.paste(img, (r * w, c * h))
    return collage
