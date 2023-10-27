
import pickle
import numpy as np

np.set_printoptions(precision=16)

from .fxpmath_conv1d import FxpMathConv1D
from .util import FxpUtil
from .activation_cache import ActivationCache

K = 4
VERBOSE = False

class FxpModel(object):

    def __init__(self, weights_file):

        with open(weights_file, 'rb') as f:
            self.weights = pickle.load(f)

        weight_ids = list(self.weights.keys())
        assert sorted(weight_ids) == weight_ids, "review what to do if this assumption breaks..."

        for weight_id in weight_ids:
            assert weight_id.startswith('qconv_'), weight_id

        self.num_layers = len(weight_ids)
        print("|layers|", self.num_layers)

        # scan each conv to derive in/out size
        in_ds = {}
        out_ds = {}
        for weight_id in weight_ids:
            weights = self.weights[weight_id]['weights'][0]
            num_kernels, in_d, out_d = weights.shape
            print("weight_id", weight_id, "num_kernels, in_d, out_d", num_kernels, in_d, out_d)
            in_ds[weight_id] = in_d
            out_ds[weight_id] = out_d
            assert num_kernels == 4
        print(f"in_ds={in_ds} & out_ds={out_ds}")
        if in_ds[weight_ids[0]] != out_ds[weight_ids[-1]]:
            raise Exception(f"expected first layer in_d to be same as last layer out_d"
                            f" but was in_ds={in_ds} out_ds={out_ds}")

        # general fxp util
        self.fxp = FxpUtil()

        # buffer for layer0 input
        self.input = np.zeros((K, in_ds[weight_ids[0]]), dtype=np.float32)

        self.qconvs = []
        self.activation_caches = []

        for layer_number, weight_id in enumerate(weight_ids):
            assert weight_id.startswith('qconv_qb'), "TODO: support qconv_po2!!"
            self.qconvs.append(FxpMathConv1D(
                self.fxp,
                weights=self.weights[weight_id]['weights'][0],
                biases=self.weights[weight_id]['weights'][1]
                ))
            is_last_layer = layer_number == self.num_layers - 1
            if not is_last_layer:
                self.activation_caches.append(ActivationCache(
                    depth=out_ds[weight_id],
                    dilation=K**(layer_number+1),
                    kernel_size=K
                ))

    def under_and_overflow_counts(self):
        return {
            'num_underflows': sum([q.num_underflows for q in self.qconvs]),
            'num_overflows': sum([q.num_overflows for q in self.qconvs])
        }

    def predict(self, x):

        # convert to near fixed point numbers and back to floats
        x = self.fxp.nparray_to_fixed_point_floats(x)
        if VERBOSE:
            print("==============")
            print("next_x", list(x))

        # shift input values left, and add new entry to idx -1
        for i in range(K-1):
            self.input[i] = self.input[i+1]
        self.input[K-1] = x
        if VERBOSE: print("lsb", self.input)

        y_pred = self.input

        for layer_id in range(self.num_layers):
            if VERBOSE: print("layer_id", layer_id)
            is_last_layer = layer_id == self.num_layers - 1
            if not is_last_layer:
                y_pred = self.qconvs[layer_id].apply(y_pred, relu=True)
                if VERBOSE: print("qconv", layer_id, "y_pred", list(y_pred))
                self.activation_caches[layer_id].add(y_pred)
                y_pred = self.activation_caches[layer_id].cached_dilated_values()
                #if VERBOSE: print("post activation_cache y_pred", list(y_pred))
            else:
                y_pred = self.qconvs[layer_id].apply(y_pred, relu=False)
                if VERBOSE: print("qconv", layer_id, "y_pred", list(y_pred))

        if VERBOSE: print("y_pred", list(y_pred))
        return y_pred



