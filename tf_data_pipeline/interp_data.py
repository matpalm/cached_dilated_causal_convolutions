import pandas as pd
import numpy as np
import random
import tensorflow as tf

# NOTE! these are ordered so that a walk of s -> r -> q -> z -> s
#       only changes one coord at a time.
wave_to_embed_pt = {
    's': [-1, -1],
    'r': [-1, 1],
    'q': [1, 1],
    'z': [1, -1]
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
        # self.WARNED = False

    def _random_interpolation_from_sample(self, sample):

        # choose alpha
        alpha = random.random()

        # interpolate samples
        # we interpolate from first output wave to second        )
        # TODO: constant power cross fade would be better here, otherwise we're getting
        #       an amplitude drop
        interpolated_samples = ( alpha * sample[:, 1]) + ((1-alpha) * sample[:, 2])

        # smooth with rolling average; can have some sharp boundaries
        # note: we need to pad to restore length ( do so with first element )
        N = 10
        interpolated_samples = moving_average(interpolated_samples, n=N)
        interpolated_samples = np.concatenate([[interpolated_samples[0]] * (N-1), interpolated_samples])

        # interpolate the embed points with the same alphas
        embed_pt_col1 = np.array(wave_to_embed_pt[self.waves[1]])
        embed_pt_col2 = np.array(wave_to_embed_pt[self.waves[2]])
        interpolated_embed_pt = ( alpha * embed_pt_col1) + ((1-alpha) * embed_pt_col2)

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

    def _first_output_wave(self, seq_len):
        # return x, y where y is the first of the outputs
        # e.g. for wave tsrq_df_w we'd return (triangle, sine)

        first_wave_id = self.waves[1]
        e0, e1 = wave_to_embed_pt[first_wave_id]

        max_offset = len(self.df) - seq_len - 1
        random_offset = random.randint(0, max_offset)
        sample = self.df[random_offset:(random_offset+seq_len)]

        x = np.zeros((len(sample), self.pad_to_size), dtype=float)
        x[:, 0] = sample[:, 0] * self.rescaling_factor
        x[:, 1] = e0 * self.rescaling_factor
        x[:, 2] = e1 * self.rescaling_factor
        y = np.zeros((len(sample), self.pad_to_size), dtype=float)
        y[:, 0] = sample[:, 1] * self.rescaling_factor

        return x, y

    def as_tf_dataset(self, seq_len, max_samples, first_output_column=False):

        def gen():
            emitted_samples = 0
            while True:
                if first_output_column:
                    yield self._first_output_wave(seq_len)
                    emitted_samples += 1
                else:
                    # if not self.WARNED:
                    #     print("WARNING! just random_interpolation=False")
                    #     self.WARNED = True
                    yield self.sample(random_interpolation=True, seq_len=seq_len)
                    yield self.sample(random_interpolation=False, seq_len=seq_len)
                    emitted_samples += 1
                if emitted_samples >= max_samples:
                    return

        return tf.data.Dataset.from_generator(
                gen, output_signature=(
                    tf.TensorSpec(shape=(seq_len, self.pad_to_size), dtype=tf.float32),
                    tf.TensorSpec(shape=(seq_len, self.pad_to_size), dtype=tf.float32)))

class Embed2DInterpolatedWaveFormData(object):

    # NOTE: the wave datas here are picked for the property that
    #       output col 1 and 2 "walk" through the space
    #       sine -> ramp -> square -> zigzag -> sine

    # NOTE: we don't use col 3 at all. had an experiment where we did, i.e. 3 way
    #       interpolation, but was too noisy based on how the data was being collected

    def __init__(self,
                root_dir,
                rescaling_factor=1,
                pad_size=8):

        tsrq_df_w, tsrq_df_l = parse(
            f"{root_dir}/tri_sine_ramp_square.ssv", 'sine', 'ramp', 'square')
        trqz_df_w, trqz_df_l = parse(
            f"{root_dir}/tri_ramp_square_zigzag.ssv", 'ramp', 'square', 'zigzag')
        tqzs_df_w, tqzs_df_l = parse(
            f"{root_dir}/tri_square_zigzag_sine.ssv", 'square', 'zigzag', 'sine')
        tzsr_df_w, tzsr_df_l = parse(
            f"{root_dir}/tri_zigzag_sine_ramp.ssv", 'zigzag', 'sine', 'ramp')

        self.wave_datas = [
            WaveData(['t', 's', 'r', 'q'], tsrq_df_w.to_numpy(),
                pad_to_size=pad_size, rescaling_factor=rescaling_factor),
            WaveData(['t', 'r', 'q', 'z'], trqz_df_w.to_numpy(),
                pad_to_size=pad_size, rescaling_factor=rescaling_factor),
            WaveData(['t', 'q', 'z', 's'], tqzs_df_w.to_numpy(),
                pad_to_size=pad_size, rescaling_factor=rescaling_factor),
            WaveData(['t', 'z', 's', 'r'], tzsr_df_w.to_numpy(),
                pad_to_size=pad_size, rescaling_factor=rescaling_factor)
        ]

    def tf_dataset_for_split(self, split, seq_len, max_samples, waves=None):

        specific_wave = None
        if waves is not None:
            if len(waves) != 1:
                raise Exception("Embed2DInterpolatedWaveFormData only support 1 wave in waves")
            specific_wave = waves[0]
            assert specific_wave in ['sine', 'ramp', 'square', 'zigzag']

        # TODO: these are actually splits at all, we just use the split to
        #       denote how to shuffle / batch

        if specific_wave is None:
            # no specific waveform requested, just output samples from all
            sampled_ds = tf.data.Dataset.sample_from_datasets(
                [wd.as_tf_dataset(seq_len, max_samples) for wd in self.wave_datas]
            )
        else:
            # a specific waveform was requested, pick the right wave data to return just that
            # ( since all wavedatas have a different first output)
            wave_data_for_specific_wave = None
            if specific_wave == 'sine':
                wave_data_for_specific_wave = self.wave_datas[0]  # tsrq_df_w
            elif specific_wave == 'ramp':
                wave_data_for_specific_wave = self.wave_datas[1]  # trqz_df_w
            elif specific_wave == 'square':
                wave_data_for_specific_wave = self.wave_datas[2]  # tqzs_df_w
            elif specific_wave == 'zigzag':
                wave_data_for_specific_wave = self.wave_datas[3]  # tzsr_df_w
            else:
                raise Exception()

            sampled_ds = wave_data_for_specific_wave.as_tf_dataset(
                seq_len, max_samples, first_output_column=True)


        if split == 'train':
            sampled_ds = sampled_ds.shuffle(1000)

        sampled_ds = sampled_ds.batch(1 if split=='test' else 64)

        return sampled_ds.prefetch(tf.data.AUTOTUNE)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import os

    data = Embed2DInterpolatedWaveFormData(rescaling_factor=1)
    tf_data = data.tf_dataset_for_split(
        split='train',
        seq_len=1000,
        max_samples=50)
        #waves=['ramp'])

    for x, y in tf_data:
        print(x.shape, y.shape)
        break

    for i in range(len(x)):

        print("i", i, x[i,0,:4])

        e0, e1 = float(x[i,0,1]), float(x[i,0,2])
        #print("e0, e1", e0, e1)

        output_dir = f"/tmp/di/" #{e0} {e1}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("PLOT")
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(x[i])
        ax2.plot(y[i])
        ax1.set_ylim((-1, 1))
        ax2.set_ylim((-1, 1))
        plt.savefig(f"{output_dir}/{i:03d}.png")
        plt.close()
