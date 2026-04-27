import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tf_data_pipeline.interp_data import Embed2DInterpolatedWaveFormData
import tqdm
import util
import warnings
import json

from qkeras_version.qkeras_model import QKerasModelBuilder

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--wave', type=str, default=None,
    help='single wave to test, if not set, test all')
parser.add_argument('--data-root-dir', type=str, required=True)
parser.add_argument("--data-rescaling-factor", type=float, default=1.953125)
parser.add_argument('--num-layers', type=int, default=4)
parser.add_argument('--filter-size', type=int, required=True)
parser.add_argument('--po2-filter-size', type=int, default=None)
parser.add_argument('--load-weights', type=str, required=True)
parser.add_argument('--test-seq-len', type=int, default=100)
opts = parser.parse_args()
print("opts", opts)

data = Embed2DInterpolatedWaveFormData(
    root_dir=opts.data_root_dir,
    rescaling_factor=opts.data_rescaling_factor,
    pad_size=4,
    seed=123)

# all convolutions use K=4
K = 4

# note: kernel size and implied dilation rate always assumed K
RECEPTIVE_FIELD_SIZE = K**opts.num_layers
TEST_SEQ_LEN = RECEPTIVE_FIELD_SIZE
print("RECEPTIVE_FIELD_SIZE", RECEPTIVE_FIELD_SIZE)
print("TEST_SEQ_LEN", TEST_SEQ_LEN)

# construct model
builder = QKerasModelBuilder()
test_model = builder.create_dilated_model(
    opts.test_seq_len,
    in_out_d=4,
    num_layers=opts.num_layers,
    filter_size=opts.filter_size,
    po2_filter_size=opts.po2_filter_size,  # if None, don't use po2
    l2=None,
)
test_model.summary()
test_model.load_weights(opts.load_weights)

# load a test set using sine wave, we'll clobber the
# embedding points so doesn't matter what this is..
test_ds = data.tf_dataset_for_split('test',
                    seq_len=opts.test_seq_len,
                    max_samples=1,
                    specific_wave='sine')
for x, _y in test_ds:
    x = np.array(x)
    break

GRID_SIZE = 7
assert GRID_SIZE%2 != 0

for i0, e0 in enumerate(np.linspace(-1, 1, GRID_SIZE)):
    e0 *= opts.data_rescaling_factor

    for i1, e1 in enumerate(np.linspace(-1, 1, GRID_SIZE)):
        e1 *= opts.data_rescaling_factor

        print("i", i0, i1, "=> e", e0, e1)

        x[0,:,1] = e0
        x[0,:,2] = e1

        y_pred = test_model.predict(x)

        # axis 0 ; just take first element ( single batch )
        # axis 1 ; drop first receptive field items ( warm up )
        # axis 2 ; just first element ( single dim output )
        y_pred = y_pred[0, 64:, 0]

        # save plot
        df = pd.DataFrame()
        df['n'] = range(opts.test_seq_len-64)
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
