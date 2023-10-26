import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import tensorflow as tf
import warnings


def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)

class CheckYPred(tf.keras.callbacks.Callback):

    def __init__(self, tb_dir, dataset, model):
        self.summary_writer = tf.summary.create_file_writer(tb_dir)
        self.model = model

        for x, y in dataset:
            self.x = x
            self.y_true = y
            break  # just one batch

        print("CheckYPred x.shape", x.shape, "y.shape", y.shape)

        # tb pagination dft is 12
        self.num_imgs_to_plot = min(12, len(self.x))

    def _plot_as_numpy(self, x, y_true, y_pred):
        df = pd.DataFrame()
        df['x'] = x[:,0]
        df['e0'] = x[:,1]
        df['e1'] = x[:,2]
        df['y_true'] = y_true[:,0]
        df['y_pred'] = y_pred[:,0]
        df['n'] = range(len(x))
        wide_df = pd.melt(df, id_vars=['n'],
                          value_vars=['x', 'y_pred', 'y_true', 'e0', 'e1'])
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            p = sns.lineplot(wide_df, x='n', y='value', hue='variable')
            p.set_ylim((-2, 2))
            # TODO better to do this direct with bytesio object...
            plt_fname = f"/dev/shm/__{random.random()}.png"
            plt.savefig(plt_fname)
            plt.clf()
        pil_img = Image.open(plt_fname).convert('RGB')
        os.remove(plt_fname)
        return np.array(pil_img)

    def on_epoch_end(self, epoch, logs=None):
        with self.summary_writer.as_default():
            with tf.name_scope("validation") as scope:
                y_pred = self.model(self.x)
                imgs = []
                for i in range(self.num_imgs_to_plot):
                    imgs.append(self._plot_as_numpy(
                        self.x[i], self.y_true[i], y_pred[i]))
                imgs = np.stack(imgs)
                tf.summary.image("check_ypred", imgs,
                    max_outputs=self.num_imgs_to_plot, step=epoch)
