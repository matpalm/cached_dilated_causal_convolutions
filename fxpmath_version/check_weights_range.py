import pickle
import argparse
import os

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--file', type=str)
parser.add_argument('--dir', type=str)
opts = parser.parse_args()
print("opts", opts)

if not((opts.file is None) ^ (opts.dir is None)):
    raise Exception("need to set one of --file or --dir")

def write_weights_range(f):
    overall_min = 1e6
    overall_max = -1e6
    all_weights = pickle.load(open(f, 'rb'))
    for conv_id in all_weights.keys():
        #print("conv_id", conv_id)
        weights = all_weights[conv_id]['weights'][0]
        biases = all_weights[conv_id]['weights'][0]
        w_min, w_max = weights.min(), weights.max()
        b_min, b_max = biases.min(), biases.max()
        #print("weight range", w_min, w_max)
        #print("bias range", b_min, b_max)
        overall_min = min([overall_min, w_min, b_min])
        overall_max = max([overall_max, w_max, b_max])
    print(f, overall_min, overall_max)

if opts.file is not None:
    write_weights_range(opts.file)
elif opts.dir is not None:
    for fname in sorted(os.listdir(opts.dir)):
        write_weights_range(os.path.join(opts.dir, fname))
else:
    raise Exception()