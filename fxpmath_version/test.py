import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import warnings
import pickle
import json

from fxpmath_version.fxpmath_model import FxpModel
from tf_data_pipeline.quadrature_data import Embed2DQuadratureData, Waveform

from . import util

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--wave",
    type=Waveform,
    default=None,
    help="single wave to test, if not set, test all",
)
# parser.add_argument('--data-root-dir', type=str, required=True)
# parser.add_argument('--data-rescaling-factor', type=float, default=1.953125)
parser.add_argument("--min-note", type=str, default="A2")
parser.add_argument("--max-note", type=str, default="A4")
parser.add_argument("--fp-int", type=int, default=4)
parser.add_argument("--fp-frac", type=int, default=12)
parser.add_argument('--load-weights', type=str)
parser.add_argument('--layer-info', type=str)
parser.add_argument('--test-x-dir', type=str, default=".")
parser.add_argument('--plot-dir', type=str, default=".")
# parser.add_argument('--write-verilog-weights', type=str,
#                     help='if set, export verilog weights')
parser.add_argument('--num-test-egs', type=int, default=100)
parser.add_argument('--verbose', action='store_true')
opts = parser.parse_args()
print("opts", opts)

# parse layer info
with open(opts.layer_info, 'r') as f:
    layer_info = json.load(f)
print("layer_info", layer_info)

# double check params
if opts.verbose and opts.wave is None:
    raise Exception(
        "need to set a --wave if --verbose ; otherwise output will clobber."
    )

# run through fxp_model
fxp_model = FxpModel(
    weights_file=opts.load_weights,
    layer_info=layer_info,
    verbose=opts.verbose)

# export weights if requested
# if opts.write_verilog_weights is not None:
#     fxp_model.export_weights_for_verilog(root_dir=opts.write_verilog_weights)

print(f"|layers|={fxp_model.num_layers()} |dilated_layers|={fxp_model.num_dilated_layers()}")

K = 4
RECEPTIVE_FIELD_SIZE = K**(fxp_model.num_dilated_layers() + 1)
# Generate enough samples to discard warmup and still evaluate num_test_egs points.
TEST_SEQ_LEN = RECEPTIVE_FIELD_SIZE + opts.num_test_egs
print("RECEPTIVE_FIELD_SIZE", RECEPTIVE_FIELD_SIZE)
print("TEST_SEQ_LEN", TEST_SEQ_LEN)

data = Embed2DQuadratureData(
    min_note=opts.min_note,
    max_note=opts.max_note,
    sample_rate_khz=192,
    fp_int=opts.fp_int,
    fp_frac=opts.fp_frac,
    seed=123,
)

fxp = util.FxpUtil()

# None:
#    util.ensure_dir_exists(opts.test_x_dir)


def process(wave):

    test_ds = data.tf_dataset(
        batch_size=16,
        seq_len=TEST_SEQ_LEN,
        num_samples=1,
        emit_specific_wave=wave,
    )

    for x, y_true in test_ds:
        x, y_true = x[0].numpy(), y_true[0].numpy()
        assert x.shape == (TEST_SEQ_LEN, fxp_model.in_dim), x.shape
        assert y_true.shape == (TEST_SEQ_LEN, fxp_model.out_dim), y_true.shape
        break

    # also write to file, if configured
    # test_x_hex_f = None
    # if opts.test_x_dir is not None:
    #    util.ensure_dir_exists(opts.test_x_dir)
    #    print("opts.test_x_dir", opts.test_x_dir)
    #     fname = f"{opts.test_x_dir}/test_x.{wave}.hex"
    #     # print("writing to", fname)
    #     # test_x_hex_f = open(fname, 'w')
    # else:
    #     print("not writing test_x.W.hex")

    # run net
    y_pred = []
    for i in tqdm.tqdm(range(len(x)), desc=f"{wave:20s}"):

        # run through model
        y_pred.append(fxp_model.predict(x[i]))

        # write data suitable for amaranth test harness

        # also write to hex file suitable for verilog tb
        # in0 in1 in2 in3
        # if test_x_hex_f is not None:
        #     hex_outputs = []
        #     for j in range(3):
        #         next_x_fp = fxp.single_width(x[i, j])
        #         next_x_fp_bits = next_x_fp.bin()
        #         next_x_fp_hex = f"0x{int(next_x_fp_bits, 2):04x}"
        #         hex_outputs.append(next_x_fp_hex)
        #     hex_outputs.append("0x0000")  # just for completeness of 4 inputs in general
        #     print(" ".join(hex_outputs), file=test_x_hex_f)
    y_pred = np.stack(y_pred)

    # Ignore the initial receptive field samples; these are affected by causal zero-padding.
    valid_start_idx = RECEPTIVE_FIELD_SIZE
    x_eval = x[valid_start_idx:]
    y_true_eval = y_true[valid_start_idx:]
    y_pred_eval = y_pred[valid_start_idx:]

    print(wave, fxp_model.under_and_overflow_counts())

    output_data_pkl_fname = os.path.join(opts.test_x_dir, wave.value, "x_yp_yt.pkl")
    util.ensure_dir_exists_for_file(output_data_pkl_fname)
    print(
        "writing x",
        x.shape,
        "y_true",
        y_true.shape,
        "y_pred",
        y_pred.shape,
        "to",
        output_data_pkl_fname,
    )
    with open(output_data_pkl_fname, "wb") as f:
        result = {
            "x": x,
            "y_true": y_true,
            "y_pred": y_pred,
            "valid_start_idx": valid_start_idx,
            "x_eval": x_eval,
            "y_true_eval": y_true_eval,
            "y_pred_eval": y_pred_eval,
        }
        pickle.dump(result, f)

    # save plot
    df = pd.DataFrame()
    df["phase_sin"] = x_eval[:, 0]
    df["phase_cos"] = x_eval[:, 1]
    df["y_pred"] = y_pred_eval[:, 0]
    df["y_true"] = y_true_eval[:, 0]
    df["n"] = range(len(y_pred_eval))
    wide_df = pd.melt(
        df,
        id_vars=["n"],
        value_vars=["phase_sin", "phase_cos", "y_pred", "y_true"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        p = sns.lineplot(wide_df, x='n', y='value', hue='variable')
        p.set(ylim=(-2, 2))
        plt_fname = f"{opts.plot_dir}/fxp_math.y_pred.{wave.value}.png"
        util.ensure_dir_exists_for_file(plt_fname)
        print("saving plot to", plt_fname)
        plt.savefig(plt_fname)
        plt.clf()


from multiprocessing import Pool

waves = [w.value for w in Waveform]
if opts.wave is None:
    p = Pool(len(waves))
    p.map(process, waves)
else:
    process(opts.wave)
