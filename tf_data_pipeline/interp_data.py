import pandas as pd
import numpy as np
import tensorflow as tf
import random

def wave_to_embed_pt(w):
    return {
        'sine': np.array([-1, -1]),
        'ramp': np.array([-1, 1]),
        'square': np.array([1, 1]),
        'zigzag': np.array([1, -1])
    }[w]

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def parse(fname, w0n, w1n, w2n):
    df_w = pd.read_csv(fname, sep=' ', names=['tri', w0n, w1n, w2n])
    df_w['n'] = range(len(df_w))
    #df_l = df_w.melt(id_vars='n', value_vars=['tri', w0n, w1n, w2n])
    return df_w.to_numpy() #, df_l

class WaveData(object):

    def __init__(self, wave0, wave1, data, pad_to_size, rescaling_factor):
        assert data.shape[1] == 3  # triangle + 2 waves
        self.wave0 = wave0
        self.wave1 = wave1
        self.embed_pt0 = wave_to_embed_pt(wave0)
        self.embed_pt1 = wave_to_embed_pt(wave1)
        self.data = data
        self.pad_to_size = pad_to_size
        self.rescaling_factor = rescaling_factor

    def sample(self, alpha, seq_len):

        # sample rows
        max_offset = len(self.data) - seq_len - 1
        random_offset = random.randint(0, max_offset)
        sample = self.data[random_offset:(random_offset+seq_len)]

        # interpolate sample
        interpolated_sample = (alpha * sample[:, 1]) + ((1-alpha) * sample[:, 2])

        # smooth with rolling average; can have some sharp boundaries
        # note: we need to pad to restore length ( do so with first element )
        N = 10
        interpolated_sample = moving_average(interpolated_sample, n=N)
        interpolated_sample = np.concatenate([[interpolated_sample[0]] * (N-1), interpolated_sample])

        # interpolate the embed points with the same alphas
        interpolated_embed_pt = ( alpha * self.embed_pt0) + ((1-alpha) * self.embed_pt1)
        interpolated_e0, interpolated_e1 = interpolated_embed_pt

        x = np.zeros((len(sample), self.pad_to_size), dtype=float)
        x[:, 0] = sample[:, 0] * self.rescaling_factor
        x[:, 1] = interpolated_e0 * self.rescaling_factor
        x[:, 2] = interpolated_e1 * self.rescaling_factor

        y = np.zeros((len(sample), self.pad_to_size), dtype=float)
        y[:, 0] = interpolated_sample * self.rescaling_factor

        return x, y

    def as_tf_dataset(self, seq_len, max_samples):

        def gen():
            num_samples_emitted = 0
            while True:
                yield self.sample(alpha=0.0, seq_len=seq_len)
                yield self.sample(alpha=1.0, seq_len=seq_len)
                #yield self.sample(alpha=random.random(), seq_len=seq_len)
                num_samples_emitted += 3
                if num_samples_emitted > max_samples:
                    return

        return tf.data.Dataset.from_generator(
            gen, output_signature=(
                tf.TensorSpec(shape=(seq_len, self.pad_to_size), dtype=tf.float32),
                tf.TensorSpec(shape=(seq_len, self.pad_to_size), dtype=tf.float32)))


class Embed2DInterpolatedWaveFormData(object):

    def __init__(self,
                root_dir,
                rescaling_factor=1,
                pad_size=8):

        print("JUST ALPHA 0.0 & 1.0, just tzs")

        tsrq_a = parse(f"{root_dir}/tri_sine_ramp_square.ssv", 'sine', 'ramp', 'square')
        tqzs_a = parse(f"{root_dir}/tri_square_zigzag_sine.ssv", 'square', 'zigzag', 'sine')

        self.tsr = WaveData('sine', 'ramp',   tsrq_a[:,[0,1,2]],
                            pad_to_size=pad_size, rescaling_factor=rescaling_factor)
        self.trq = WaveData('ramp', 'square', tsrq_a[:,[0,2,3]],
                            pad_to_size=pad_size, rescaling_factor=rescaling_factor)
        self.tqz = WaveData('square', 'zigzag', tqzs_a[:,[0,1,2]],
                            pad_to_size=pad_size, rescaling_factor=rescaling_factor)
        self.tzs = WaveData('zigzag', 'sine', tqzs_a[:,[0,2,3]],
                            pad_to_size=pad_size, rescaling_factor=rescaling_factor)

        self.all_wave_data = [ self.tsr, self.trq, self.tqz, self.tzs ]
        #self.all_wave_data = [ self.tsr, self.trq, self.tqz ]

    def tf_dataset_for_split(self, split, seq_len, max_samples, waves=None):

        assert waves == None, "TODO"

        sampled_ds = tf.data.Dataset.sample_from_datasets(
            [wd.as_tf_dataset(seq_len, max_samples) for wd in self.all_wave_data]
        )

        if split == 'train':
            sampled_ds = sampled_ds.shuffle(1000)

        sampled_ds = sampled_ds.batch(1 if split=='test' else 64)

        return sampled_ds.prefetch(tf.data.AUTOTUNE)