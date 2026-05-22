import os
import io

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

    def __init__(self, tb_dir, dataset):
        self.summary_writer = tf.summary.create_file_writer(tb_dir)

        for x, y in dataset:
            self.x = x
            self.y_true = y
            break  # just one batch

    def _plot_as_numpy(self, x, y_true, y_pred):
        df = pd.DataFrame()
        df["phase_sin"] = x[:, 0]
        df["e0"] = x[:, 2]
        df["e1"] = x[:, 3]
        df['y_true'] = y_true[:,0]
        df['y_pred'] = y_pred[:,0]
        df['n'] = range(len(x))
        wide_df = pd.melt(
            df, id_vars=["n"], value_vars=["phase_sin", "y_pred", "y_true", "e0", "e1"]
        )
        with io.BytesIO() as img_buffer:
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                plt.figure(figsize=(14, 4))
                p = sns.lineplot(wide_df, x="n", y="value", hue="variable")
                p.set_ylim((-2, 2))
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format="png")
                plt.clf()
            img_buffer.seek(0)
            pil_img = Image.open(img_buffer).convert("RGB")
        return np.array(pil_img)

    def on_epoch_end(self, epoch, logs=None):
        with self.summary_writer.as_default():
            with tf.name_scope("validation") as scope:
                y_pred = self.model(self.x)

                # tb pagination dft is 12, so take at most 2 pages
                plot_x = self.x[:24]
                y_pred = y_pred[:24]
                plot_y_true = self.y_true[:24]

                # never show more than 1000 samples in time
                # and if trimming, pick the last
                # plot_x = plot_x[:, -1000:]
                # y_pred = y_pred[:, -1000:]
                # plot_y_true = plot_y_true[:, -1000:]

                imgs = []
                for i in range(len(plot_x)):
                    imgs.append(
                        self._plot_as_numpy(plot_x[i], plot_y_true[i], y_pred[i])
                    )
                imgs = np.stack(imgs)
                tf.summary.image(
                    "check_ypred", imgs, max_outputs=len(plot_x), step=epoch
                )
