from fxpmath_version.fxpmath_model import FxpModel
import util
import os

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--weights', type=str)
opts = parser.parse_args()
print("opts", opts)

fxp_model = FxpModel(opts.weights)

util.ensure_dir_exists('weights')

for i, qconv_layer in enumerate(fxp_model.qconvs):
    fname = f"weights/qconv{i}"
    print("exporting qconv", i, "to", fname)
    qconv_layer.export_weights_for_verilog(fname)
