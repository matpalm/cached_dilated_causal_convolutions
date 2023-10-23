from fxpmath_version.fxpmath_model import FxpModel
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from tf_data_pipeline.data import WaveToWaveData, Embed2DWaveFormData
from tf_data_pipeline.interp_data import Embed2DInterpolatedWaveFormData
import tqdm
import util
import warnings

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--wave', type=str, default=None,
    help='single wave to test, if not set, test all')
parser.add_argument('--data-root-dir', type=str, required=True)
parser.add_argument('--data-rescaling-factor', type=float, default=1.953125)
parser.add_argument('--in-out-d-filter-size', type=int, default=8)
parser.add_argument('--load-weights', type=str)
parser.add_argument('--write-test-x', action='store_true')
parser.add_argument('--write-verilog-weights', type=str,
                    help='if set, export verilog weights')
parser.add_argument('--num-test-egs', type=int, default=100)
opts = parser.parse_args()
print("opts", opts)

# run through fxp_model
fxp_model = FxpModel(opts.load_weights)

# export weights if requested
if opts.write_verilog_weights is not None:
    for i, qconv_layer in enumerate(fxp_model.qconvs):
        fname = f"{opts.write_verilog_weights}/qconv{i}"
        print("exporting qconv", i, "to", fname)
        qconv_layer.export_weights_for_verilog(fname)

IN_OUT_D = FILTER_SIZE = 8

print("|layers|=", fxp_model.num_layers)
K = 4
RECEPTIVE_FIELD_SIZE = K**fxp_model.num_layers
TEST_SEQ_LEN = RECEPTIVE_FIELD_SIZE
print("RECEPTIVE_FIELD_SIZE", RECEPTIVE_FIELD_SIZE)
print("TEST_SEQ_LEN", TEST_SEQ_LEN)

data = Embed2DInterpolatedWaveFormData(
    root_dir=opts.data_root_dir,
    rescaling_factor=opts.data_rescaling_factor,
    pad_size=opts.in_out_d_filter_size)

fxp = util.FxpUtil()

def process(wave):
    print("running wave", wave)

    test_ds = data.tf_dataset_for_split('test',
                        seq_len=opts.num_test_egs,
                        max_samples=1,
                        specific_wave=wave)

    for x, y in test_ds:
        x, y = x[0].numpy(), y[0].numpy()
        IN_OUT_D = x.shape[1]
        assert x.shape == (opts.num_test_egs, IN_OUT_D), x.shape
        assert y.shape == (opts.num_test_egs, IN_OUT_D), y.shape
        break

    # also write to file, if configured
    test_x_hex_f = None
    if opts.write_test_x:
        fname = f"test_x.{wave}.hex"
        print("writing to", fname)
        test_x_hex_f = open(fname, 'w')
    else:
        print("not writing test_x.W.hex")

    # run net
    y_pred = []
    for i in tqdm.tqdm(range(len(x))):

        # run through model
        y_pred.append(fxp_model.predict(x[i]))

        # also write to hex file suitable for verilog tb
        # in0 in1 in2 in3
        if test_x_hex_f is not None:
            hex_outputs = []
            for j in range(3):
                next_x_fp = fxp.single_width(x[i, j])
                next_x_fp_bits = next_x_fp.bin()
                next_x_fp_hex = f"0x{int(next_x_fp_bits, 2):04x}"
                hex_outputs.append(next_x_fp_hex)
            hex_outputs.append("0x0000")  # just for completeness of 4 inputs in general
            print(" ".join(hex_outputs), file=test_x_hex_f)

    print(wave, fxp_model.under_and_overflow_counts())

    y_pred = np.stack(y_pred)

    if test_x_hex_f is not None:
        test_x_hex_f.close()

    # save plot
    df = pd.DataFrame()
    df['y_pred'] = y_pred[:,0]
    df['y_true'] = y[:,0]
    df['n'] = range(len(y_pred))
    wide_df = pd.melt(df, id_vars=['n'], value_vars=['y_pred', 'y_true'])
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        p = sns.lineplot(wide_df, x='n', y='value', hue='variable')
        p.set(ylim=(-2, 2))
        plt_fname = f"fxp_math.y_pred.{wave}.png"
        print("saving plot to", plt_fname)
        plt.savefig(plt_fname)
        plt.clf()

from multiprocessing import Pool
waves = ['sine', 'ramp', 'square', 'zigzag']
if opts.wave is None:
    p = Pool(len(waves))
    p.map(process, waves)
else:
    assert opts.wave in waves
    process(opts.wave)

