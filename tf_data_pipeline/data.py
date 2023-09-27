import pandas as pd
import numpy as np
import random
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

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

def extract_x_y(data, x2, x3, data_axis_for_y):
    result = {}
    result['x'] = np.empty((len(data), 3), dtype=np.float32)
    result['x'][:,0] = x2
    result['x'][:,1] = x3
    result['x'][:,2] = data[:, 0] # triangle
    result['y'] = np.expand_dims(data[:,data_axis_for_y], -1) # target wave
    return result

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

    def __init__(self, root_dir='datalogger_firmware/data', generate_plots=False):
        tsr_df_w, tsr_df_l = parse(f"{root_dir}/2d_embed/32kHz/tri_sine_ramp.ssv")
        tsz_df_w, tsz_df_l = parse(f"{root_dir}/2d_embed/32kHz/tri_squ_zigzag.ssv")

        # # sanity check plot
        if generate_plots:
            plot_train_data(tsr_df_l, "triangle_sine_ramp.train.png")
            plot_train_data(tsz_df_l, "triangle_square_zigzag.train.png")

        # rebuild datasets as tri_to[WAVE][X/Y] with triangle wave in x with x2, x3
        # and target wave as y
        self.tri_to = {}
        data = tsr_df_w.to_numpy().astype(np.float32)
        self.tri_to['sine'] = extract_x_y(data, x2=0, x3=0, data_axis_for_y=1)
        self.tri_to['ramp'] = extract_x_y(data, x2=0, x3=1, data_axis_for_y=2)
        data = tsz_df_w.to_numpy().astype(np.float32)
        self.tri_to['square'] = extract_x_y(data, x2=1, x3=0, data_axis_for_y=1)
        self.tri_to['zigzag'] = extract_x_y(data, x2=1, x3=1, data_axis_for_y=2)

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
