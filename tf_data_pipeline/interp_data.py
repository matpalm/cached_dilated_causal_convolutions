import pandas as pd
import numpy as np
import random
import tensorflow as tf

wave_to_embed_pt = {
    's': [-1, -1],
    'r': [-1, 1],
    'z': [1, 1],
    'q': [1, -1]
}

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def parse(fname, w0n, w1n, w2n):
    df_w = pd.read_csv(fname, sep=' ', names=['tri', w0n, w1n, w2n])
    df_w['n'] = range(len(df_w))
    df_l = df_w.melt(id_vars='n', value_vars=['tri', w0n, w1n, w2n])
    return df_w, df_l


class WaveData(object):

    def __init__(self, waves, df, pad_to_size, rescaling_factor):
        self.waves = waves
        self.wave_to_column = { name: idx for idx, name in enumerate(waves) }
        self.df = df
        self.pad_to_size = pad_to_size
        self.rescaling_factor = rescaling_factor

    def _random_interpolation_from_sample(self, sample):

        interp_waves = self.waves[1:]
        random.shuffle(interp_waves)
#        print("interp_waves", interp_waves)

        # choose alpha for interps
        alpha1 = random.random()
        alpha2 = random.random()
#        print("alphas", alpha1, alpha2)

        # interpolate samples
        # TODO: constant power cross fade would be better here, otherwise we're getting
        #       an amplitude drop
        interp_columns = [self.wave_to_column[w] for w in interp_waves]
#        print("interp_columns", interp_columns)
        interpolated_samples = ( alpha1 * sample[:, interp_columns[0]]) + ((1-alpha1) * sample[:, interp_columns[1]])
        interpolated_samples = ( alpha2 * sample[:, interp_columns[2]]) + ((1-alpha2) * interpolated_samples)

        # smooth with rolling average; can have some sharp boundaries
        # note: we need to pad to restore length ( do so with first element )
        N = 10
        interpolated_samples = moving_average(interpolated_samples, n=N)
        interpolated_samples = np.concatenate([[interpolated_samples[0]] * (N-1), interpolated_samples])

        # interpolate the embed points with the same alphas
        embed_pts = [np.array(wave_to_embed_pt[w]) for w in interp_waves]
#        print("interp_pts", interp_pts)
        interpolated_embed_pt = ( alpha1 * embed_pts[0]) + ((1-alpha1) * embed_pts[1])
#        print("interpolated_embed_pt", interpolated_embed_pt)
        interpolated_embed_pt = ( alpha2 * embed_pts[2]) + ((1-alpha2) * interpolated_embed_pt)
#        print("interpolated_embed_pt", interpolated_embed_pt)

        e0 = interpolated_embed_pt[0]  # e0
        e1 = interpolated_embed_pt[1]  # e1
        sy = interpolated_samples

        return e0, e1, sy


    def _random_single_wave(self, sample):

        random_wave = random.choice(self.waves[1:])
        random_wave_col = self.wave_to_column[random_wave]
        embed_pt = wave_to_embed_pt[random_wave]

        e0 = embed_pt[0]   # e0
        e1 = embed_pt[1]   # e1
        sy = sample[:, random_wave_col]

        return e0, e1, sy


    def sample(self, random_interpolation, seq_len):

        max_offset = len(self.df) - seq_len - 1
        random_offset = random.randint(0, max_offset)
        sample = self.df[random_offset:(random_offset+seq_len)]

        if random_interpolation:
            # interpolate randomly between the three waves
            e0, e1, sy = self._random_interpolation_from_sample(sample)
        else:
            # just pick one of the waves to emit
            e0, e1, sy = self._random_single_wave(sample)

        x = np.zeros((len(sample), self.pad_to_size), dtype=float)
        x[:, 0] = sample[:, 0] * self.rescaling_factor
        x[:, 1] = e0 * self.rescaling_factor
        x[:, 2] = e1 * self.rescaling_factor
        y = np.zeros((len(sample), self.pad_to_size), dtype=float)
        y[:, 0] = sy * self.rescaling_factor

        return x, y

class Embed2DInterpolatedWaveFormData(object):

    def __init__(self,
                root_dir='datalogger_firmware/data/2d_embed_interp/48kHz/',
                rescaling_factor=1):

        tsrq_df_w, tsrq_df_l = parse(
            f"{root_dir}/tri_sine_ramp_square.ssv", 'sine', 'ramp', 'square')
        trqz_df_w, trqz_df_l = parse(
            f"{root_dir}/tri_ramp_square_zigzag.ssv", 'ramp', 'square', 'zigzag')
        tqzs_df_w, tqzs_df_l = parse(
            f"{root_dir}/tri_square_zigzag_sine.ssv", 'square', 'zigzag', 'sine')
        tzsr_df_w, tzsr_df_l = parse(
            f"{root_dir}/tri_zigzag_sine_ramp.ssv", 'zigzag', 'sine', 'ramp')

        tsrq_w = tsrq_df_w.to_numpy()
        trqz_w = trqz_df_w.to_numpy()
        tqzs_w = tqzs_df_w.to_numpy()
        tzsr_w = tzsr_df_w.to_numpy()

        self.wave_datas = [
            WaveData(['t', 's', 'r', 'q'], tsrq_w, pad_to_size=8, rescaling_factor=rescaling_factor),
            WaveData(['t', 'r', 'q', 'z'], trqz_w, pad_to_size=8, rescaling_factor=rescaling_factor),
            WaveData(['t', 'q', 'z', 's'], tqzs_w, pad_to_size=8, rescaling_factor=rescaling_factor),
            WaveData(['t', 'z', 's', 'r'], tzsr_w, pad_to_size=8, rescaling_factor=rescaling_factor)
        ]

    def make_dataset(self, seq_len):

        def gen():
            while True:
                for wd in self.wave_datas:
                    yield wd.sample(random_interpolation=True, seq_len=seq_len)
                    yield wd.sample(random_interpolation=False, seq_len=seq_len)


        return tf.data.Dataset.from_generator(
                gen, output_signature=(
                    tf.TensorSpec(shape=(seq_len, 8), dtype=tf.float32),
                    tf.TensorSpec(shape=(seq_len, 8), dtype=tf.float32)))



if __name__ == '__main__':

    data = Embed2DInterpolatedWaveFormData(rescaling_factor=20)
    tf_data = data.make_dataset(seq_len=1000).batch(20)

    for x, y in tf_data:
        print(x.shape, y.shape)
        break

    import matplotlib.pyplot as plt

    for i in range(len(x)):
        fig, (ax1, ax2) = plt.subplots(2)
        print(i, x[i][:5])
        ax1.plot(x[i])
        ax2.plot(y[i])
        ax1.set_ylim((-21, 21))
        ax2.set_ylim((-21, 21))
        plt.savefig(f"/tmp/data_interp_{i:03d}.png")
        fig.clf()
