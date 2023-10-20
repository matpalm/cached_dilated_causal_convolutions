import pickle
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--weights', type=str)
opts = parser.parse_args()
print("opts", opts)

overall_min = 1e6
overall_max = -1e6

all_weights = pickle.load(open(opts.weights, 'rb'))
for conv_id in all_weights.keys():
    print("conv_id", conv_id)
    weights = all_weights[conv_id]['weights'][0]
    biases = all_weights[conv_id]['weights'][0]
    w_min, w_max = weights.min(), weights.max()
    b_min, b_max = biases.min(), biases.max()
    print("weight range", w_min, w_max)
    print("bias range", b_min, b_max)
    overall_min = min([overall_min, w_min, b_min])
    overall_max = max([overall_max, w_max, b_max])

print("overall_min", overall_min, "overall_max", overall_max)