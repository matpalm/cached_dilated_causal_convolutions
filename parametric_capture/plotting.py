# seaborn just wont shut up
import warnings
import io
from typing import List

warnings.simplefilter(action="ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import librosa
from PIL import Image

from .audio_interface import SAMPLE_RATE_HZ


def fig_as_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")
    pil_img.load()  # or we loss it via close
    buf.close()
    plt.close(fig)  # otherwise notebook shows _two_ plots inline :/
    return pil_img


def plot(
    array,
    title: str = "",
    ch_names: List[str] = None,
    fname: str = None,
    plot_offset: int = 20_000,
    plot_len: int = 2_000,
):
    if len(array.shape) == 1:
        array = array.reshape((-1, 1))

    if plot_offset is not None:
        array = array[plot_offset : plot_offset + plot_len]

    _, c = array.shape
    fig, axes = plt.subplots(c, 1, figsize=(12, 3 * c), sharex=True)
    axes = np.atleast_1d(axes)

    for i in range(c):
        ch_name = ch_names[i] if ch_names is not None else f"ch{i}"
        ch_name += f" {title}"
        data = array[:, i]
        axes[i].plot(data)
        axes[i].set_title(ch_name)
        axes[i].set_ylabel("Amplitude")
        axes[i].set_ylim(-1.0, 1.0)
    axes[-1].set_xlabel("sample")

    fig.tight_layout()
    if fname is None:
        return fig_as_pil(fig)
    fig.savefig(fname, dpi=500)
    plt.close(fig)


def plot_spectrogram(
    series,
    title: str = "spectrogram",
    fname: str = None,
    plot_offset: int = None,
    plot_len: int = None,
    max_freq_hz: float = 6_000,
):
    # NFFT=4096 gives ~2.9 Hz bins at 12 kHz, resolving 5 Hz spacing
    series = np.asarray(series)
    if plot_offset is not None:
        series = series[plot_offset : plot_offset + plot_len]

    nfft = 4096
    noverlap = nfft * 3 // 4
    hop_length = 256

    # Generate an STFT spectrogram in dB using librosa, then render with matplotlib.
    stft = librosa.stft(series.astype(np.float32), n_fft=nfft, hop_length=hop_length)
    spec_db = librosa.power_to_db(np.abs(stft) ** 2, ref=np.max)

    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE_HZ, n_fft=nfft)
    times = librosa.frames_to_time(
        np.arange(spec_db.shape[1]),
        sr=SAMPLE_RATE_HZ,
        hop_length=hop_length,
    )

    if times.size == 0:
        time_extent_max = 0.0
    else:
        time_extent_max = times[-1]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        cmap="inferno",
        extent=[0.0, time_extent_max, freqs[0], freqs[-1]],
        interpolation="nearest",
    )
    ax.set_ylim(0, max_freq_hz)
    ax.set_yticks(np.arange(0, max_freq_hz + 1, 5))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)

    fig.tight_layout()
    if fname is None:
        return fig_as_pil(fig)
    fig.savefig(fname, dpi=500)
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
