from fxpmath_version.fxpmath_model import FxpModel
import util
import os

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input-weights-pkl', type=str)
parser.add_argument('--output-weights-dir', type=str)
opts = parser.parse_args()
print("opts", opts)

fxp_model = FxpModel(opts.input_weights_pkl, verbose=False)
fxp_model.export_weights_for_verilog(root_dir=opts.output_weights_dir)

