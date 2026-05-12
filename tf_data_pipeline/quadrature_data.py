import numpy as np
from enum import Enum

FREQS = {"A2": 110, "A3": 220, "A4": 440, "A5": 880, "A6": 1760}

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
                return np.where(phase_cos >= 0.0, 1.0, -1.0)
            case Waveform.SINE:
                # Cosine peaks at phase 0, aligning with triangle/ramp peaks.
                return phase_cos
            case Waveform.RAMP:
                # Descending ramp with peak at phase 0.
                return 1.0 - (2.0 * cycle)

    if interp == 0:
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
        result = soft_clip(result)
        # and rescale back to (-1, 1)

    return {
        "phase_sin": phase_sin,
        "phase_cos": phase_cos,
        "wave": result,
    }


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import seaborn as sns

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--min-note", type=str, default="A3")
    parser.add_argument("--max-note", type=str, default="A5")
    parser.add_argument("--sample-rate-hz", type=float, default=192 * 1000)
    parser.add_argument("--starting-phase", type=float, default=0)
    parser.add_argument("--num_samples", type=int, default=1000)
    opts = parser.parse_args()
    print("opts", opts)

    # all waves

    # x = None
    # for wave in [Waveform.TRIANGLE, Waveform.SQUARE, Waveform.RAMP, Waveform.SINE]:
    #     data = calculate_interp_wave(
    #         opts.frequency_hz,
    #         opts.sample_rate_hz,
    #         opts.starting_phase,
    #         opts.num_samples,
    #         wave,
    #         None,
    #         interp=0,
    #     )

    #     if x is None:
    #         x = np.arange(len(data["phase_sin"]))
    #         sns.set_theme(style="whitegrid")
    #         fig, (ax_top, ax_bottom) = plt.subplots(
    #             2,
    #             1,
    #             sharex=True,
    #             figsize=(12, 8),
    #             constrained_layout=True,
    #         )
    #         sns.lineplot(x=x, y=data["phase_sin"], ax=ax_top, label="phase_sin")
    #         sns.lineplot(x=x, y=data["phase_cos"], ax=ax_top, label="phase_cos")
    #         ax_top.set_ylabel("phase")
    #         ax_top.set_title("Quadrature Phase")

    #     sns.lineplot(x=x, y=data["wave"], ax=ax_bottom, label=str(wave))

    # ax_bottom.set_ylabel("amplitude")
    # ax_bottom.set_xlabel("sample index")
    # ax_bottom.set_title("single waves")

    # fig.savefig("single_waves.png")

    # interp wave egs
    import random

    rng = random.Random(123)

    for i in range(100):
        frequency_hz = sample_freq(FREQS["A2"], FREQS["A4"], alpha=rng.random())
        starting_phase = rng.random() * 2 * np.pi
        all_waves = [Waveform.TRIANGLE, Waveform.SQUARE, Waveform.RAMP, Waveform.SINE]
        w1 = rng.choice(all_waves)
        w2 = w1
        while w2 == w1:
            w2 = rng.choice(all_waves)
        interp = rng.random()

        data_w1 = calculate_wave(
            frequency_hz,
            opts.sample_rate_hz,
            starting_phase,
            opts.num_samples,
            w1,
        )

        data_w2 = calculate_wave(
            frequency_hz,
            opts.sample_rate_hz,
            starting_phase,
            opts.num_samples,
            w2,
        )

        data_interp = calculate_wave(
            frequency_hz,
            opts.sample_rate_hz,
            starting_phase,
            opts.num_samples,
            w1,
            w2,
            interp,
        )

        x = np.arange(len(data_w1["phase_sin"]))

        # sns.lineplot(x=x, y=data[["phase_sin"], ax=ax_top, label="phase_sin")
        # sns.lineplot(x=x, y=data[0]["phase_cos"], ax=ax_top, label="phase_cos")
        # ax_top.set_ylabel("phase")
        # ax_top.set_title("Quadrature Phase")
        plt.clf()
        sns.lineplot(x=x, y=data_w1["wave"], label=f"w1={w1.value}")
        sns.lineplot(x=x, y=data_w2["wave"], label=f"w2={w2.value}")
        sns.lineplot(x=x, y=data_interp["wave"], label=f"interp={interp:0.2f}")
        plt.title(f"frequency_hz={frequency_hz:.2f}")
        plt.savefig(f"interp_data_egs/{i:04d}.png")
