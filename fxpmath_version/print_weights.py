import pickle
import argparse
import os

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--file', type=str)
parser.add_argument('--dir', type=str)
opts = parser.parse_args()

if not((opts.file is None) ^ (opts.dir is None)):
    raise Exception("need to set one of --file or --dir")

def write_weights_range(f):
    all_weights = pickle.load(open(f, 'rb'))
    for conv_id in all_weights.keys():
        weights = all_weights[conv_id]['weights'][0]
        biases = all_weights[conv_id]['weights'][1]
        for w in weights.flatten():
            print(w)
        for b in biases.flatten():
            print(b)

if opts.file is not None:
    write_weights_range(opts.file)
elif opts.dir is not None:
    for fname in sorted(os.listdir(opts.dir)):
        write_weights_range(os.path.join(opts.dir, fname))
else:
    raise Exception()