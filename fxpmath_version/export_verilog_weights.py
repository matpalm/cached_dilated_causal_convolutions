from fxpmath_version.fxpmath_model import FxpModel
import os
import json

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input-weights-pkl', type=str)
parser.add_argument('--layer-info', type=str)
parser.add_argument('--output-weights-dir', type=str)
opts = parser.parse_args()
print("opts", opts)

with open(opts.layer_info, 'r') as f:
    layer_info = json.load(f)

fxp_model = FxpModel(
    weights_file=opts.input_weights_pkl,
    layer_info=layer_info,
    verbose=False)

fxp_model.export_weights_for_verilog(root_dir=opts.output_weights_dir)

