import numpy as np
import random
from scipy.stats import qmc

from .audio_interface import SAMPLE_RATE_HZ

def generate_cv_schroeder_multisine(
    freq_spacing_hz: float = 500,
    frequencies_per_output: int = 4,
    len_sec: float = 2.0,
    num_orthogonal: int = 1,
):
    """Generate num_orthogonal multisines split over interleaved freq bins.

    e.g. num_orthogonal=1, frequencies_per_output=4, freq_spacing_hz=500
    generates one multisine with freqs (500, 1000, 1500, 2000)

    e.g. num_orthogonal=3, frequencies_per_output=3, freq_spacing_hz=100
    generates three multisines
    multisine_0 with freqs (100, 400, 700)
    multisine_1 with freqs (200, 500, 800)
    multisine_2 with freqs (300, 600, 900)
    """

    total_num_frequencies = frequencies_per_output * num_orthogonal

    t = np.linspace(0, len_sec, int(SAMPLE_RATE_HZ * len_sec), endpoint=False)
    signals = np.zeros((num_orthogonal, len(t)), dtype=float)

    for i in range(num_orthogonal):
        signal = np.zeros_like(t)
        for k in range(i + 1, total_num_frequencies + 1, num_orthogonal):
            phase = (np.pi * (k**2)) / total_num_frequencies
            f_k = k * freq_spacing_hz
            signal += np.cos(2 * np.pi * f_k * t + phase)
        max_abs = np.max(np.abs(signal))
        if max_abs > 0:
            signal /= max_abs
        signals[i] = signal

    return signals


class SineWaveSampler(object):

    def __init__(
        self,
        min_freq_hz: float,
        max_freq_hz: float,
        seed: int = 123,
    ):
        self.rng = random.Random(seed)
        self.min_freq_hz = min_freq_hz
        self.diff_freq_hz = max_freq_hz - min_freq_hz

    def sample(self, sample_len_s: float):
        starting_phase = self.rng.random() * 2 * np.pi
        freq_hz = self.min_freq_hz + (self.rng.random() * self.diff_freq_hz)
        phase_step = 2.0 * np.pi * (freq_hz / SAMPLE_RATE_HZ)
        sample_len = sample_len_s * SAMPLE_RATE_HZ
        phase = starting_phase + (phase_step * np.arange(sample_len))
        return np.sin(phase)


class SobolSampler(object):

    def __init__(
        self,
        bounds,  # list of 2 tuples
        seed: int,
    ):
        for b in bounds:
            if len(b) != 2:
                raise Exception("bounds should be list of 2 tuples")
        self.lower_bounds, self.upper_bounds = list(zip(*bounds))
        num_d = len(self.lower_bounds)
        self.sampler = qmc.Sobol(d=num_d, scramble=True, seed=seed)

    def samples(self, num_samples_po2: int, fast_forward: int = None):
        # note: doesnt have to be po2
        if fast_forward:
            self.sampler.fast_forward(fast_forward)
        samples = self.sampler.random(n=num_samples_po2)  # (num_samples, num_d) (0, 1)
        return qmc.scale(samples, self.lower_bounds, self.upper_bounds)
