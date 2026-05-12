import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tf_data_pipeline.quadrature_data import Embed2DQuadratureData, Waveform
import tqdm
import warnings
import json

from qkeras_version.qkeras_model import QKerasModelBuilder

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--min-note", type=str, default="A2")
parser.add_argument("--max-note", type=str, default="A4")
parser.add_argument(
    "--fp-int",
    type=int,
    default=4,
    help=" integer bits for FP config",
)
parser.add_argument(
    "--fp-frac",
    type=int,
    default=12,
    help="fractional bits for FP config",
)
parser.add_argument("--filter-sizes", type=int, nargs="+", required=True)
parser.add_argument("--relu-upper-bound", type=float, default=6)
parser.add_argument("--load-weights", type=str, required=True)
parser.add_argument(
    "--wave", type=str, default=None, help="single wave to test, if not set, test all"
)
parser.add_argument("--test-seq-len", type=int, default=400)
opts = parser.parse_args()
print("opts", opts)

data = Embed2DQuadratureData(
    min_note=opts.min_note,
    max_note=opts.max_note,
    seed=123,
)

# all convolutions use K=4
K = 4
num_layers = len(opts.filter_sizes)

# note: kernel size and implied dilation rate always assumed K
RECEPTIVE_FIELD_SIZE = K**num_layers
print("RECEPTIVE_FIELD_SIZE", RECEPTIVE_FIELD_SIZE)
print("TEST_SEQ_LEN", opts.test_seq_len)

# construct model
builder = QKerasModelBuilder(n_int=opts.fp_int, n_frac=opts.fp_frac)
test_model = builder.create_dilated_model(
    opts.test_seq_len,
    in_out_d=4,
    filter_sizes=opts.filter_sizes,
    l2=None,
    relu_upper_bound=opts.relu_upper_bound,
)
test_model.summary()
test_model.load_weights(opts.load_weights)

# load a test set using sine wave, we'll clobber the
# embedding points so doesn't matter what this is..
test_ds = data.tf_dataset(
    batch_size=16,
    seq_len=opts.test_seq_len,
    num_samples=1,
    emit_specific_wave=opts.wave,
)
for x, _y in test_ds:
    x = np.array(x)
    break

GRID_SIZE = 7
assert GRID_SIZE%2 != 0

for i0, e0 in enumerate([-1]):  # np.linspace(-1, 1, GRID_SIZE)):

    for i1, e1 in enumerate([-1]):  # np.linspace(-1, 1, GRID_SIZE)):

        print("i", i0, i1, "=> e", e0, e1)
        x[0, :, 2] = e0
        x[0, :, 3] = e1

        y_pred = test_model.predict(x)

        # axis 0 ; just take first element ( single batch )
        # axis 1 ; drop first receptive field items ( warm up )
        # axis 2 ; just first element ( single dim output )
        y_pred = y_pred[0, RECEPTIVE_FIELD_SIZE:, 0]

        # save plot
        df = pd.DataFrame()
        df["n"] = range(len(y_pred))
        df['y_pred'] = y_pred
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            p = sns.lineplot(df, x='n', y='y_pred', linewidth=5)
            p.set(xticklabels=[])
            p.set(xlabel=None)
            p.set(yticklabels=[])
            p.set(ylabel=None)
            p.tick_params(bottom=False, left=False)
            p.set(ylim=(-2, 2))
            plt_fname = f"foo_{i0:02d}_{i1:02d}.png"
            print("saving plot to", plt_fname)
            plt.savefig(plt_fname)
            plt.clf()
