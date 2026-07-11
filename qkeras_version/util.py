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
        assert y_true.shape == y_pred.shape
        assert y_true.shape[-1] == 1

        x = np.asarray(x, dtype=np.float32)
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)

        df = pd.DataFrame()
        df["triangle"] = x[:, 0]
        df["a_cv"] = x[:, 1]
        df["b_cv"] = x[:, 2]
        df["morph_cv"] = x[:, 3]
        df["y_true"] = y_true[:, 0]
        df["y_pred"] = y_pred[:, 0]
        df['n'] = range(len(x))
        control_df = pd.melt(
            df,
            id_vars=["n"],
            value_vars=["triangle", "a_cv", "b_cv", "morph_cv"],
        )
        output_df = pd.melt(
            df,
            id_vars=["n"],
            value_vars=["y_true", "y_pred"],
        )
        with io.BytesIO() as img_buffer:
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                fig, axes = plt.subplots(
                    2,
                    1,
                    figsize=(20, 6),
                    sharex=True,
                    gridspec_kw={"height_ratios": [2, 1]},
                )
                sns.lineplot(control_df, x="n", y="value", hue="variable", ax=axes[0])
                axes[0].set_ylim((-2, 2))
                axes[0].set_ylabel("control")
                sns.lineplot(output_df, x="n", y="value", hue="variable", ax=axes[1])
                axes[1].set_ylim((-2, 2))
                axes[1].set_ylabel("output")
                axes[1].set_xlabel("n")
                fig.tight_layout()
                fig.savefig(img_buffer, format="png")
                plt.close(fig)
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
