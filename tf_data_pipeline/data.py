import pandas as pd
import numpy as np
import random
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

raise Exception("SWITCH TO INTERP!")

def parse(fname):
    df_w = pd.read_csv(fname, sep=' ', names=['tri', 'w0', 'w1'])
    df_w['n'] = range(len(df_w))
    df_l = df_w.melt(id_vars='n', value_vars=['tri', 'w0', 'w1'])
    return df_w, df_l

def plot_train_data(df, fname):
    plt.clf()
    plt.figure(figsize=(16, 6))
    offset = 5500
    width = 1000
    window = (df['n']>offset) & (df['n']<offset+width)
    sns.lineplot(df[window], x='n', y='value', hue='variable')
    plt.savefig(fname)

def extract_x_y(data, e0, e1, data_axis_for_y):
    x = np.zeros((len(data), 8), dtype=np.float32)
    x[:,0] = data[:, 0] # triangle
    x[:,1] = e0
    x[:,2] = e1
    # => x[:,3] = 0

    y = np.zeros((len(data), 8), dtype=np.float32)
    y[:,0] = data[:,data_axis_for_y]

    return {'x': x, 'y': y}

def split_train_val_test(d):
    assert 'x' in d
    assert 'y' in d
    assert len(d['x']) == len(d['y'])
    val_test_split_size = int(len(d['x']) * 0.1)  # 10% for val and test
    d['train'] = {}
    d['validate'] = {}
    d['test'] = {}
    for xy in ['x', 'y']:
        d['train'][xy] = d[xy][:-2*val_test_split_size]
        d['validate'][xy] = d[xy][-2*val_test_split_size:-val_test_split_size]
        d['test'][xy] = d[xy][-val_test_split_size:]
        d.pop(xy)

def tf_dataset_from(x, y, split, max_samples, seq_len):
    in_d, out_d = x.shape[-1], y.shape[-1]

    def gen():
        idxs = list(range(len(x)-seq_len-1))  # ~1.3M
        if split != 'test':
            random.Random(1337).shuffle(idxs)
        idxs = idxs[:max_samples]
        for i in idxs:
            yield x[i:i+seq_len], y[i+1:i+1+seq_len]

    return tf.data.Dataset.from_generator(
        gen, output_signature=(tf.TensorSpec(shape=(seq_len, in_d), dtype=tf.float32),
                               tf.TensorSpec(shape=(seq_len, out_d), dtype=tf.float32)))



class Embed2DWaveFormData(object):

    def __init__(self,
                    root_dir='datalogger_firmware/data/2d_embed/32kHz/',
                    generate_plots=False,
                    rescaling_factor=1):

        tsr_df_w, tsr_df_l = parse(f"{root_dir}/tri_sine_ramp.ssv")
        tsz_df_w, tsz_df_l = parse(f"{root_dir}/tri_squ_zigzag.ssv")

        # # sanity check plot
        if generate_plots:
            plot_train_data(tsr_df_l, "triangle_sine_ramp.train.png")
            plot_train_data(tsz_df_l, "triangle_square_zigzag.train.png")

        # rebuild datasets as tri_to[WAVE][X/Y] with triangle wave in x with e0,e1
        # and target wave as y
        RS = rescaling_factor
        self.tri_to = {}
        data = tsr_df_w.to_numpy().astype(np.float32)
        data *= rescaling_factor
        self.tri_to['sine'] = extract_x_y(data, e0=0, e1=0, data_axis_for_y=1)
        self.tri_to['ramp'] = extract_x_y(data, e0=0, e1=RS, data_axis_for_y=2)
        data = tsz_df_w.to_numpy().astype(np.float32)
        data *= rescaling_factor
        self.tri_to['square'] = extract_x_y(data, e0=RS, e1=0, data_axis_for_y=1)
        self.tri_to['zigzag'] = extract_x_y(data, e0=RS, e1=RS, data_axis_for_y=2)

        # train/val/test split
        for wave in ['sine', 'ramp', 'square', 'zigzag']:
            split_train_val_test(self.tri_to[wave])

    def tf_dataset_for_split(self,
            split, seq_len, max_samples,
            waves=['sine', 'ramp', 'square', 'zigzag']):
        # dataset that is input (triangle, C1, C2) -> (output waves based on C1, C2)

        assert split in ['train', 'validate', 'test']
        sampled_ds = tf.data.Dataset.sample_from_datasets([
            tf_dataset_from(
                self.tri_to[wave][split]['x'],
                self.tri_to[wave][split]['y'],
                split, max_samples, seq_len)
            for wave in waves
        ])

        if split == 'train':
            sampled_ds = sampled_ds.shuffle(1000)

        sampled_ds = sampled_ds.batch(1 if split=='test' else 128)

        return sampled_ds.prefetch(tf.data.AUTOTUNE)


class WaveToWaveData(object):

    def __init__(self,
                 root_dir='datalogger_firmware/data/2d_embed/32kHz/',
                 in_out_d=8,
                 rescaling_factor=1  # as a workaround to calibration on FPGA
                 ):

        print("WaveToWave root_dir", root_dir)

        fname = f"{root_dir}/tri_squ_zigzag.ssv"
        data = pd.read_csv(fname, sep=' ', names=['tri', 'square', 'zigzag']).to_numpy()

        # for prototype
        # TDO: change this to output square in first slot
        n = len(data)
        x = np.concatenate([data[:,0:1], np.zeros((n, in_out_d-1))], axis=-1)  # (tri, 0, 0, 0, 0, 0, 0, 0)
        y = np.concatenate([data, np.zeros((n, in_out_d-3))], axis=-1)         # (tri, square, zigzag, 0, 0, 0, 0, 0)
        assert x.shape == (n, in_out_d)
        assert y.shape == (n, in_out_d)

        if rescaling_factor != 1:
            x *= rescaling_factor
            y *= rescaling_factor

        # take splits
        val_test_split_size = int(len(x) * 0.1)  # 10% for val and test
        self.data = {
            'train': { 'x': x[:-2*val_test_split_size],
                       'y': y[:-2*val_test_split_size]},
            'validate': { 'x': x[-2*val_test_split_size:-val_test_split_size],
                          'y': y[-2*val_test_split_size:-val_test_split_size]},
            'test': { 'x': x[-val_test_split_size:],
                      'y': y[-val_test_split_size:]}
        }

    def tf_dataset_for_split(self, split, seq_len, max_samples):
        x = self.data[split]['x']
        y = self.data[split]['y']

        in_d, out_d = x.shape[-1], y.shape[-1]

        def gen():
            idxs = list(range(len(x)-seq_len-1))
            if split != 'test':
                random.Random(1337).shuffle(idxs)
            idxs = idxs[:max_samples]
            for i in idxs:
                yield x[i:i+seq_len], y[i+1:i+1+seq_len]

        ds = tf.data.Dataset.from_generator(
            gen, output_signature=(tf.TensorSpec(shape=(seq_len, in_d), dtype=tf.float32),
                                   tf.TensorSpec(shape=(seq_len, out_d), dtype=tf.float32)))

        if split == 'train':
            ds = ds.shuffle(1000)

        ds = ds.batch(1 if split=='test' else 128)

        return ds.prefetch(tf.data.AUTOTUNE)


class Embed2DInterpData(object):

    def __init__(self,
                    root_dir='datalogger_firmware/data/2d_embed/32kHz/',
                    generate_plots=False,
                    rescaling_factor=1):

        tsr_df_w, tsr_df_l = parse(f"{root_dir}/tri_sine_ramp.ssv")
        tsz_df_w, tsz_df_l = parse(f"{root_dir}/tri_squ_zigzag.ssv")

        # # sanity check plot
        if generate_plots:
            plot_train_data(tsr_df_l, "triangle_sine_ramp.train.png")
            plot_train_data(tsz_df_l, "triangle_square_zigzag.train.png")

        # rebuild datasets as tri_to[WAVE][X/Y] with triangle wave in x with e0,e1
        # and target wave as y
        RS = rescaling_factor
        self.tri_to = {}
        data = tsr_df_w.to_numpy().astype(np.float32)
        data *= rescaling_factor
        self.tri_to['sine'] = extract_x_y(data, e0=0, e1=0, data_axis_for_y=1)
        self.tri_to['ramp'] = extract_x_y(data, e0=RS, e1=RS, data_axis_for_y=2)
        data = tsz_df_w.to_numpy().astype(np.float32)
        data *= rescaling_factor
        self.tri_to['square'] = extract_x_y(data, e0=RS, e1=0, data_axis_for_y=1)
        self.tri_to['zigzag'] = extract_x_y(data, e0=0, e1=RS, data_axis_for_y=2)

        # train/val/test split
        for wave in ['sine', 'ramp', 'square', 'zigzag']:
            split_train_val_test(self.tri_to[wave])

    def tf_dataset_for_split(self,
            split, seq_len, max_samples,
            waves=['sine', 'ramp', 'square', 'zigzag']):
        # dataset that is input (triangle, C1, C2) -> (output waves based on C1, C2)

        assert split in ['train', 'validate', 'test']
        sampled_ds = tf.data.Dataset.sample_from_datasets([
            tf_dataset_from(
                self.tri_to[wave][split]['x'],
                self.tri_to[wave][split]['y'],
                split, max_samples, seq_len)
            for wave in waves
        ])

        if split == 'train':
            sampled_ds = sampled_ds.shuffle(1000)

        sampled_ds = sampled_ds.batch(1 if split=='test' else 128)

        return sampled_ds.prefetch(tf.data.AUTOTUNE)

# ds = WaveToWaveData(in_out_d=6, rescaling_factor=5.0)
# for x, y in ds.tf_dataset_for_split('test', seq_len=4, max_samples=10):
#     print(x, y)
#     break