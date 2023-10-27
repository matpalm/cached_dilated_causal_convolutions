import pickle
import argparse
import numpy as np
import os
from .util import FxpUtil

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--file', type=str)
parser.add_argument('--dir', type=str)
opts = parser.parse_args()

fxp_util = FxpUtil()

if not((opts.file is None) ^ (opts.dir is None)):
    raise Exception("need to set one of --file or --dir")

def print_elements(a, a_str_id):
    it = np.nditer(a, flags=['multi_index'])
    for v in it:
        fp_v = fxp_util.single_width(v)
        safe_log2 = "-" if v == 0 else np.log2(abs(v))
        print("\t".join(map(str, [a_str_id, it.multi_index, v, fp_v, safe_log2])))

def write_weights_range(f):
    all_weights = pickle.load(open(f, 'rb'))
    for conv_id in all_weights.keys():
        print_elements(
            all_weights[conv_id]['weights'][0],  # weights
            f"{conv_id}_w"
        )
        print_elements(
            all_weights[conv_id]['weights'][1],  # biases
            f"{conv_id}_b"
        )

print("\t".join(['a_str_id', 'idx', 'v', 'fp_v', 'fp_safe_log2']))

if opts.file is not None:
    write_weights_range(opts.file)
elif opts.dir is not None:
    for fname in sorted(os.listdir(opts.dir)):
        write_weights_range(os.path.join(opts.dir, fname))
else:
    raise Exception()