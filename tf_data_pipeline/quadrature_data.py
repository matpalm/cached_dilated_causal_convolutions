import numpy as np
from enum import Enum
import random
import tensorflow as tf

FREQS = {"A2": 110, "A3": 220, "A4": 440, "A5": 880, "A6": 1760}

IN_OUT_D = 4

class Waveform(Enum):
    TRIANGLE = "triangle"
    SQUARE = "square"
    SINE = "sine"
    RAMP = "ramp"

    def to_embed_pt(self):
        return {
            Waveform.TRIANGLE: np.array([1, -1]),
            Waveform.SQUARE: np.array([1, 1]),
            Waveform.SINE: np.array([-1, -1]),
            Waveform.RAMP: np.array([-1, 1]),
        }[self]


def soft_clip(x, drive=1.5):
    x_ref = max(1.0, np.max(np.abs(x))) + 1e-10
    x_rescaled = x / x_ref
    return np.tanh(drive * x_rescaled) / np.tanh(drive)


def sample_freq(min_freq, max_freq, alpha):
    if min_freq == max_freq:
        return min_freq
    assert min_freq < max_freq
    assert 0 <= alpha <= 1
    min_freq_2 = np.log2(min_freq)
    max_freq_2 = np.log2(max_freq)
    diff = max_freq_2 - min_freq_2
    sample_freq_2 = min_freq_2 + alpha * diff
    return 2**sample_freq_2


def calculate_wave(
    frequency_hz: float,
    sample_rate_hz: float,
    starting_phase: float,
    num_samples: int,
    waveform1: Waveform,
    waveform2: Waveform = None,
    interp: float = 0,
    scale: float = 0.8,
):
    if frequency_hz > (sample_rate_hz / 2.0):
        raise ValueError("faildog! nyquist limit")

    phase_step = 2.0 * np.pi * (frequency_hz / sample_rate_hz)
    phase = starting_phase + (phase_step * np.arange(num_samples, dtype=np.float64))
    phase_sin = np.sin(phase)
    phase_cos = np.cos(phase)
    cycle = np.mod(phase / (2.0 * np.pi), 1.0)  # [0, 1)

    def wave(w):
        match w:
            case Waveform.TRIANGLE:
                # inverted c.f. others
                return (4.0 * np.abs(np.mod(cycle + 0.5, 1.0) - 0.5)) - 1.0
            case Waveform.SQUARE:
                return np.where(phase_cos >= 0.0, 1, -1)
            case Waveform.SINE:
                # Cosine peaks at phase 0, aligning with triangle/ramp peaks.
                return phase_cos
            case Waveform.RAMP:
                # Descending ramp with peak at phase 0.
                return 1.0 - (2.0 * cycle)

    if interp == 0 or waveform2 is None:
        result = wave(waveform1)
    elif interp == 1:
        result = wave(waveform2)
    else:
        # interpolate sample ( constant power cross fade interpolation)
        # 0 => w1, 1 => w2.
        result1 = wave(waveform1)
        result2 = wave(waveform2)
        s1 = np.sin((1 - interp) * np.pi / 2)
        s2 = np.sin(interp * np.pi / 2)
        result = (s1 * result1) + (s2 * result2)
        # note doesn't ensure values stay in (-1, 1) so soft clip
        # result = soft_clip(result)
        result = np.clip(result, 0, 1)

    return {
        "phase_sin": scale * phase_sin,
        "phase_cos": scale * phase_cos,
        "wave": scale * result,
    }


class Embed2DQuadratureData(object):

    def __init__(
        self,
        min_note: str,
        max_note: str,
        # sample_rate_hz: float,
        seed: int = 123,
    ):
        self.min_note = min_note
        self.max_note = max_note
        self.sample_rate_hz = 196 * 1000
        self.rng = random.Random(seed)

    def _sample_wave(self, seq_len, w1, w2=None, interp=None):

        frequency_hz = sample_freq(
            FREQS[self.min_note], FREQS[self.max_note], alpha=self.rng.random()
        )
        starting_phase = self.rng.random() * 2 * np.pi

        if w2 is None:
            interp = 0.0
        elif interp is None:
            interp = self.rng.random()

        data = calculate_wave(
            frequency_hz,
            self.sample_rate_hz,
            starting_phase,
            seq_len,
            w1,
            w2,
            interp,
        )

        if w2 is None:
            embed_pt = w1.to_embed_pt()
        else:
            embed_pt = ((1.0 - interp) * w1.to_embed_pt()) + (interp * w2.to_embed_pt())

        return data, embed_pt

    def _xy_from_data(self, data, embed_pt):
        # TODO: this could be a map in tf
        N = len(data["phase_sin"])
        x = np.zeros((N, IN_OUT_D), dtype=np.float32)
        y = np.zeros((N, IN_OUT_D), dtype=np.float32)
        x[:, 0] = data["phase_sin"]
        x[:, 1] = data["phase_cos"]
        x[:, 2] = embed_pt[0]
        x[:, 3] = embed_pt[1]
        y[:, 0] = data["wave"]
        return x, y

    def tf_dataset(
        self,
        batch_size: int,
        seq_len: int,
        num_samples: int,
        emit_endpt_samples: bool = True,
        emit_interpolated_samples: bool = True,
        emit_specific_wave: Waveform = None,
    ):

        if type(emit_specific_wave) == str:
            emit_specific_wave = Waveform(emit_specific_wave)

        def gen_waves():
            while True:
                if emit_specific_wave is not None:
                    # just emit data for this specific wave
                    yield self._sample_wave(
                        seq_len=seq_len, w1=emit_specific_wave, w2=None
                    )
                else:
                    assert emit_endpt_samples or emit_interpolated_samples
                    # samples two waves
                    w1, w2 = self.rng.choice(
                        [
                            (Waveform.RAMP, Waveform.SQUARE),
                            (Waveform.SQUARE, Waveform.TRIANGLE),
                            (Waveform.TRIANGLE, Waveform.SINE),
                            (Waveform.SINE, Waveform.RAMP),
                        ]
                    )
                    # emit either interpolated, or the two ends points
                    if emit_endpt_samples:
                        yield self._sample_wave(seq_len=seq_len, w1=w1, w2=None)
                        yield self._sample_wave(seq_len=seq_len, w1=w2, w2=None)
                    if emit_interpolated_samples:
                        yield self._sample_wave(seq_len=seq_len, w1=w1, w2=w2)

        def gen_limited_number():
            g = gen_waves()
            for _ in range(num_samples):
                data, embed_pt = next(g)
                yield self._xy_from_data(data, embed_pt)

        ds = tf.data.Dataset.from_generator(
            gen_limited_number,
            output_signature=(
                tf.TensorSpec(shape=(seq_len, IN_OUT_D), dtype=tf.float32),
                tf.TensorSpec(shape=(seq_len, IN_OUT_D), dtype=tf.float32),
            ),
        )
        #        ds = ds.shuffle(batch_size * 5)
        ds = ds.batch(batch_size)
        return ds.prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import seaborn as sns

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--min-note", type=str, default="A3")
    parser.add_argument("--max-note", type=str, default="A5")
    # parser.add_argument("--sample-rate-hz", type=float, default=192 * 1000)
    parser.add_argument("--starting-phase", type=float, default=0)
    parser.add_argument("--num_samples", type=int, default=1000)
    opts = parser.parse_args()
    print("opts", opts)
    import os

    os.makedirs("interp_data_egs", exist_ok=True)

    data_source = Embed2DQuadratureData(
        min_note=opts.min_note,
        max_note=opts.max_note,
        #        sample_rate_hz=opts.sample_rate_hz,
        seed=123,
    )

    ds = data_source.tf_dataset(
        batch_size=1,
        seq_len=opts.num_samples,
        num_samples=300,
        emit_endpt_samples=True,
        emit_interpolated_samples=True,
    )

    plot_idx = 0
    for x, y in ds.take(100):
        xs, xc = x[0, :, 0], x[0, :, 1]
        e0, e1 = x[0, :, 2], x[0, :, 3]
        yt = y[0, :, 0]
        x = np.arange(len(xs))
        plt.clf()
        sns.lineplot(x=x, y=xs, label="xs")
        sns.lineplot(x=x, y=xc, label="xc")
        sns.lineplot(x=x, y=e0, label=f"e0 {e0[0]:0.2f}")
        sns.lineplot(x=x, y=e1, label=f"e1 {e1[0]:0.2f}")
        sns.lineplot(x=x, y=yt, label="yt")
        plt.savefig(f"interp_data_egs/{plot_idx:04d}.png")
        plot_idx += 1
