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


WAVE_EDGES = [
    (Waveform.RAMP, Waveform.SQUARE),
    (Waveform.SQUARE, Waveform.TRIANGLE),
    (Waveform.TRIANGLE, Waveform.SINE),
    (Waveform.SINE, Waveform.RAMP),
]


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
    interp_start: float = 0,
    interp_end: float = 0,
    scale: float = 0.8,
):
    if frequency_hz > (sample_rate_hz / 2.0):
        raise ValueError("faildog! nyquist limit")

    phase_step = 2.0 * np.pi * (frequency_hz / sample_rate_hz)
    phase = starting_phase + (phase_step * np.arange(num_samples))
    phase_sin = np.sin(phase)
    phase_cos = np.cos(phase)
    cycle = np.mod(phase / (2.0 * np.pi), 1.0)  # [0, 1)

    def wave(w):
        match w:
            case Waveform.TRIANGLE:
                return 1.0 - (4.0 * np.abs(np.mod(cycle + 0.5, 1.0) - 0.5))
            case Waveform.SQUARE:
                return np.where(phase_sin >= 0.0, 1, -1)
            case Waveform.SINE:
                return phase_sin
            case Waveform.RAMP:
                return 1.0 - (2.0 * cycle)

    if waveform2 is None:
        result = wave(waveform1)
        interp = None
    else:
        # interp is _start for first 1/4, _end for last 1/4 and linear between
        interp = np.full(num_samples, interp_start, dtype=np.float64)
        i25 = int(0.25 * num_samples)
        i65 = int(0.65 * num_samples)
        if i65 > i25:
            interp[i25:i65] = np.linspace(interp_start, interp_end, i65 - i25)
        interp[i65:] = interp_end
        interp = np.clip(interp, 0.0, 1.0)
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
        "interp": interp,
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

    def random_freq(self):
        return sample_freq(
            FREQS[self.min_note], FREQS[self.max_note], alpha=self.rng.random()
        )

    def random_phase(self):
        return self.rng.random() * 2 * np.pi

    def _sample_single_wave(self, seq_len, w1):
        data = calculate_wave(
            self.random_freq(),
            self.sample_rate_hz,
            self.random_phase(),
            seq_len,
            w1,
            waveform2=None,
        )
        embed_pt = w1.to_embed_pt()
        return data, embed_pt

    def _sample_interpolated_wave(self, seq_len, w1, w2, interp_start, interp_end):
        data = calculate_wave(
            self.random_freq(),
            self.sample_rate_hz,
            self.random_phase(),
            seq_len,
            w1,
            w2,
            interp_start,
            interp_end,
        )
        interp = data["interp"].astype(np.float32)
        embed_pt = ((1.0 - interp)[:, None] * w1.to_embed_pt()) + (
            interp[:, None] * w2.to_embed_pt()
        )
        return data, embed_pt

    def _xy_from_data(self, data, embed_pt):
        # TODO: this could be a map in tf
        N = len(data["phase_sin"])
        x = np.zeros((N, IN_OUT_D), dtype=np.float32)
        y = np.zeros((N, IN_OUT_D), dtype=np.float32)
        x[:, 0] = data["phase_sin"]
        x[:, 1] = data["phase_cos"]
        if np.ndim(embed_pt) == 1:
            x[:, 2] = embed_pt[0]
            x[:, 3] = embed_pt[1]
        else:
            x[:, 2] = embed_pt[:, 0]
            x[:, 3] = embed_pt[:, 1]
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
                    w1, w2 = self.rng.choice(WAVE_EDGES)
                    # emit either interpolated, or the two ends points
                    if emit_endpt_samples:
                        yield self._sample_single_wave(seq_len, w1)
                        yield self._sample_single_wave(seq_len, w2)
                    if emit_interpolated_samples:
                        interp_start = self.rng.random()
                        interp_end = self.rng.random()
                        yield self._sample_interpolated_wave(
                            seq_len, w1, w2, interp_start, interp_end
                        )

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
    import pandas as pd

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

    # ds = data_source.tf_dataset(
    #     batch_size=1,
    #     seq_len=opts.num_samples,
    #     num_samples=300,
    #     emit_endpt_samples=True,
    #     emit_interpolated_samples=True,
    # )

    for w1, w2 in WAVE_EDGES:
        seq_len = 1000
        data = calculate_wave(
            frequency_hz=FREQS["A4"],
            sample_rate_hz=64_000,
            starting_phase=0,
            num_samples=seq_len,
            waveform1=w1,
            waveform2=w2,
            interp_start=0.0,
            interp_end=1.0,
        )
        df = pd.DataFrame()
        df["n"] = range(seq_len)
        df["wave"] = data["wave"]
        df["interp"] = data["interp"]
        p = sns.lineplot(df, x="n", y="wave", linewidth=5)
        p = sns.lineplot(df, x="n", y="interp", linewidth=5)
        plt.savefig(f"interp_data_egs/{w1}_{w2}.png")
        plt.clf()

    # GRID_SIZE = 5

    # for i0, e0 in enumerate(np.linspace(-1, 1, GRID_SIZE)):

    #     for i1, e1 in enumerate(np.linspace(-1, 1, GRID_SIZE)):

    #         print("i", i0, i1, "=> e", e0, e1)
    #         x[0, :, 2] = e0
    #         x[0, :, 3] = e1

    #         y_pred = test_model.predict(x)

    #         # axis 0 ; just take first element ( single batch )
    #         # axis 1 ; drop first receptive field items ( warm up )
    #         # axis 2 ; just first element ( single dim output )
    #         y_pred = y_pred[0, RECEPTIVE_FIELD_SIZE:, 0]

    #         # save plot
    #         df = pd.DataFrame()
    #         df["n"] = range(len(y_pred))
    #         df["y_pred"] = y_pred
    #         with warnings.catch_warnings():
    #             warnings.simplefilter(action="ignore", category=FutureWarning)
    #             p = sns.lineplot(df, x="n", y="y_pred", linewidth=5)
    #             p.set(xticklabels=[])
    #             p.set(xlabel=None)
    #             p.set(yticklabels=[])
    #             p.set(ylabel=None)
    #             p.tick_params(bottom=False, left=False)
    #             p.set(ylim=(-2, 2))
    #             plt_fname = f"foo_{i0:02d}_{i1:02d}.png"
    #             print("saving plot to", plt_fname)
    #             plt.savefig(plt_fname)
    #             plt.clf()
