
import pickle
import numpy as np

np.set_printoptions(precision=16)

from .fxpmath_conv1d_qb import FxpMathConv1DQuantisedBitsBlock
from .fxpmath_conv1d_po2 import FxpMathConv1DPO2Block
from .util import FxpUtil
from .activation_cache import ActivationCache

K = 4

class Relu(object):
    def __init__(self, zero_value):
        self.zero_value = zero_value

    def apply(self, x):
        for i in range(len(x)):
            if x[i] < 0:
                x[i] = self.zero_value
        return x

    def __str__(self):
        return "relu"


def check_ids_match(weights, layer_info):
    w_ids = set(weights.keys())
    l_ids = set()
    for l in layer_info:
        if 'id' in l:
            l_ids.add(l['id'])
    if w_ids != l_ids:
        raise Exception(f"mismatch between weight ids [{w_ids}] and layer ids [{l_ids}]")


class FxpModel(object):

    def __init__(self, weights_file, layer_info, verbose):

        with open(weights_file, 'rb') as f:
            self.weights = pickle.load(f)
        weight_ids = list(self.weights.keys())

        print("weight_ids", weight_ids)
        print("layer_info", layer_info)

        # check there is a layer for each weight and vice versa
        check_ids_match(self.weights, layer_info)

        self.weight_ids = weight_ids
        for weight_id in weight_ids:
            assert weight_id.startswith('qconv_')
            assert weight_id.endswith('_qb') or weight_id.endswith('_po2')

        self._num_layers = len(weight_ids)
        print("|layers|", self._num_layers)

        # scan each conv to derive in/out size
        in_ds = {}
        out_ds = {}
        for weight_id in weight_ids:
            weights = self.weights[weight_id]['weights'][0]
            num_kernels, in_d, out_d = weights.shape
            print(f"weight_id={weight_id} num_kernels={num_kernels} in_d={in_d} out_d={out_d}")
            in_ds[weight_id] = in_d
            out_ds[weight_id] = out_d
            if weight_id.endswith('_qb'):
                assert num_kernels == 4
            elif weight_id.endswith('_po2'):
                assert num_kernels in [1, 4]

        print(f"in_ds={in_ds} & out_ds={out_ds}")
        if in_ds[weight_ids[0]] != out_ds[weight_ids[-1]]:
            raise Exception(f"expected first layer in_d to be same as last layer out_d"
                            f" but was in_ds={in_ds} out_ds={out_ds}")

        # general fxp util
        self.fxp = FxpUtil()

        # make stateless relu layer
        relu = Relu(zero_value=self.fxp.double_width(0))

        # buffer for layer0 input
        self.in_dim = in_ds[weight_ids[0]]
        self.input = np.zeros((K, self.in_dim), dtype=np.float32)

        self.verbose = verbose

        self._num_dilated_layers = 0
        self.layers = []
        for info in layer_info:
            if info['type'] == 'qb':
                layer_id = info['id']
                next_layer = FxpMathConv1DQuantisedBitsBlock(
                    self.fxp,
                    layer_name=layer_id,
                    weights=self.weights[layer_id]['weights'][0],
                    biases=self.weights[layer_id]['weights'][1],
                    verbose=self.verbose
                    )
            elif info['type'] == 'po2':
                layer_id = info['id']
                next_layer = FxpMathConv1DPO2Block(
                    self.fxp,
                    layer_name=layer_id,
                    weights=self.weights[layer_id]['weights'][0],
                    biases=self.weights[layer_id]['weights'][1],
                    verbose=self.verbose
                )
            elif info['type'] == 'dilation':
                next_layer = ActivationCache(
                    depth=info['depth'],
                    dilation=info['amount'],
                    kernel_size=K
                )
                self._num_dilated_layers += 1
            elif info['type'] == 'relu':
                next_layer = relu
            else:
                raise Exception(f"unhandled type [{info['type']}]")

            print(type(next_layer), next_layer)
            self.layers.append(next_layer)

    def num_layers(self):
        return self._num_layers

    def num_dilated_layers(self):
        return self._num_dilated_layers

    def under_and_overflow_counts(self):
        num_underflows = 0
        num_overflows = 0
        for l in self.layers:
            try:
                num_underflows += l.num_overflows()
                num_overflows += l.num_overflows()
            except AttributeError:
                pass
        return {'num_underflows': num_underflows,
                'num_overflows': num_overflows }

    def predict(self, x):

        # convert to near fixed point numbers and back to floats
        x = self.fxp.nparray_to_fixed_point_floats(x)
        if self.verbose:
            print("==============")
            print("next_x", list(x))

        # shift input values left, and add new entry to idx -1
        for i in range(K-1):
            self.input[i] = self.input[i+1]
        self.input[K-1] = x
        if self.verbose: print("lsb", self.input)

        y_pred = self.input

        for layer in self.layers:
            if self.verbose: print("running layer", layer)
            y_pred = layer.apply(y_pred)
            if self.verbose: print("result ", list(y_pred))

        if self.verbose: print("y_pred", list(y_pred))
        return y_pred

    def export_weights_for_verilog(self, root_dir):
        for layer in self.layers:
            try:
                layer.export_weights_for_verilog(root_dir)
            except AttributeError:
                pass
