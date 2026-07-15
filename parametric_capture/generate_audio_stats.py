from pathlib import Path
import zarr
import numpy as np
from scipy.signal import lfilter, lfilter_zi
import argparse
import pickle
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm


def calculate_audio_stats(series, ignore_in_out: int = 500):
    """Calculate some states of an audio series

    stat                    low value       high value
    rms                     quiet           loud
    spectral centroid       "dark" / bass   "bright" / treble
    total harmonic dist     "clean"         "distorted"
    spectral flatness       "tonal" / pure  "noisy" / buzzy
    odd/even ratio          "full" (saw)    "hollow" (square)

    args:
        series: audio signal, with values (-1, 1)
        ignore_in_out: number of samples to ignore at start and end ( fade in out )
    returns:
        dictionary with {
            "rms": rms value,
            "spectral_centroid": spectral centroid,
            "total_harmonic_dist": total harmonic distortion,
            "flatness": spectral flatness (geometric / arithmetic mean),
            "odd_even": odd / even harmonic energy ratio (excludes fundamental),
            "harmonic_profile": np.array of harmonic amplitudes a_k / a_1 for
                k = 1 .. N_HARMONICS (a pitch-invariant timbre fingerprint;
                compare two via cosine distance),
        }
    """
    assert len(series.shape) == 1
    assert series.min() >= -1
    assert series.max() <= 1
    assert ignore_in_out >= 0
    assert (2 * ignore_in_out) < len(series)

    series = series[ignore_in_out:-ignore_in_out]

    rms = np.sqrt(np.mean(np.square(series)))

    # TODO: librosa probably has these plus more?

    # windowed spectrum, used for both spectral centroid and thd. the hann
    # window contains spectral leakage so that off-bin fundamentals don't smear
    # energy across the spectrum (which biases both metrics).
    window = np.hanning(len(series))
    spectrum = np.abs(np.fft.rfft(series * window))

    # spectral centroid
    sc = 0.0
    freqs = np.fft.rfftfreq(len(series), d=1.0)
    spec_sum = np.sum(spectrum)
    if spec_sum > 0:
        sc = float(np.sum(freqs * spectrum) / spec_sum)

    # total harmonic distortion ( relative to the strongest non-DC component ).
    #
    # integrate power in a small band of bins around each harmonic. without this,
    # a single-bin estimate is highly sensitive to whether the fundamental lands
    # exactly on an fft bin: spectral leakage either smears harmonic energy out
    # of the sampled bin (THD too low) or leaks fundamental energy into it (THD
    # too high), so perceptually identical waveforms at slightly different pitch
    # get very different THD.
    thd = 0.0
    power = spectrum**2
    n_bins = len(power)

    # number of harmonics ( including fundamental ) kept for the timbre profile
    N_HARMONICS = 8
    harmonic_profile = np.zeros(N_HARMONICS)
    odd_even = 0.0

    if n_bins > 2:
        fundamental_bin = int(np.argmax(power[1:]) + 1)

        # hann main lobe is ~4 bins wide, so integrate +/- a couple of bins
        band = 2

        def band_power(center):
            lo = max(1, center - band)
            hi = min(n_bins, center + band + 1)
            if lo >= n_bins:
                return 0.0
            return float(np.sum(power[lo:hi]))

        fundamental_power = band_power(fundamental_bin)
        if fundamental_power > 0:
            harmonic_power, h = 0.0, 2
            while (h * fundamental_bin) < n_bins:
                harmonic_power += band_power(h * fundamental_bin)
                h += 1
            thd = float(np.sqrt(harmonic_power) / np.sqrt(fundamental_power))

            # per-harmonic amplitudes, normalised to the fundamental. this vector
            # is a pitch-invariant timbre fingerprint: index k holds a_k / a_1.
            fundamental_amp = np.sqrt(fundamental_power)
            for k in range(1, N_HARMONICS + 1):
                bin_k = k * fundamental_bin
                if bin_k >= n_bins:
                    break
                harmonic_profile[k - 1] = np.sqrt(band_power(bin_k)) / fundamental_amp

            # odd / even harmonic energy ratio ( fundamental excluded ). a square
            # wave is odd-only ("hollow"), a saw has both ("full").
            odd_power = even_power = 0.0
            k = 2
            while k * fundamental_bin < n_bins:
                p = band_power(k * fundamental_bin)
                if k % 2:
                    odd_power += p
                else:
                    even_power += p
                k += 1
            if even_power > 0:
                odd_even = float(odd_power / even_power)

    # spectral flatness ( geometric mean / arithmetic mean of the power spectrum,
    # excluding DC ): ~0 for a pure tone, ~1 for noise / very buzzy timbres.
    flatness = 0.0
    non_dc_power = power[1:]
    if non_dc_power.size and np.all(np.isfinite(non_dc_power)):
        arithmetic_mean = float(np.mean(non_dc_power))
        if arithmetic_mean > 0:
            geometric_mean = float(np.exp(np.mean(np.log(non_dc_power + 1e-12))))
            flatness = geometric_mean / arithmetic_mean

    return {
        "rms": rms,
        "spectral_centroid": sc,
        "total_harmonic_dist": thd,
        "flatness": flatness,
        "odd_even": odd_even,
        # "harmonic_profile": harmonic_profile,
    }


if __name__ == "__main__":

    raise Exception("out of date")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True)
    opts = parser.parse_args()

    capture_buffers_z = zarr.open("runs" / opts.run / "capture_buffers.z", mode="r")

    def ch0_gen():
        for b in range(capture_buffers_z.nchunks):
            yield np.asarray(capture_buffers_z.blocks[b][:, 0])

    with Pool(processes=8) as pool:
        records = list(
            tqdm(
                pool.imap(
                    calculate_audio_stats,
                    ch0_gen(),
                    chunksize=4,
                ),
                total=capture_buffers_z.nchunks,
                desc=f"calculating audio stats for {opts.run}",
            )
        )

    with open("runs" / opts.run / "audio_stats.pkl", "wb") as f:
        pickle.dump(records, f)
